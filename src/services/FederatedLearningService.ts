import AsyncStorage from '@react-native-async-storage/async-storage';

export interface ModelUpdate {
  participantId: string;
  modelWeights: number[];
  trainingMetrics: {
    accuracy: number;
    loss: number;
    epochs: number;
    dataSize: number;
  };
  timestamp: Date;
}

export interface GlobalModel {
  version: string;
  accuracy: number;
  participants: number;
  lastUpdate: Date;
  downloadUrl?: string;
}

export interface FeedbackData {
  participantId: string;
  predictedEvent: string;
  actualEvent: string;
  confidence: number;
  audioFeatures?: number[];
}

class FederatedLearningService {
  private baseUrl: string;
  private wsUrl: string;
  private ws: WebSocket | null = null;
  private participantId: string | null = null;
  private isConnected: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private eventListeners: Map<string, Function[]> = new Map();

  constructor() {
    // Use localhost for development, replace with your production URL
    this.baseUrl = __DEV__ ? 'http://localhost:3001/api' : 'https://your-production-api.com/api';
    this.wsUrl = __DEV__ ? 'ws://localhost:3001' : 'wss://your-production-api.com';
    this.loadParticipantId();
  }

  private async loadParticipantId() {
    try {
      const stored = await AsyncStorage.getItem('fl_participant_id');
      if (stored) {
        this.participantId = stored;
      }
    } catch (error) {
      console.error('Error loading participant ID:', error);
    }
  }

  private async saveParticipantId(id: string) {
    try {
      await AsyncStorage.setItem('fl_participant_id', id);
      this.participantId = id;
    } catch (error) {
      console.error('Error saving participant ID:', error);
    }
  }

  // Event system for real-time updates
  on(event: string, callback: Function) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(callback);
  }

  off(event: string, callback: Function) {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      const index = listeners.indexOf(callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  }

  // WebSocket connection management
  async connectWebSocket(): Promise<boolean> {
    return new Promise((resolve) => {
      try {
        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = () => {
          console.log('✅ Connected to federated learning server');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.emit('connected', true);
          
          // Join federated learning if we have a participant ID
          if (this.participantId) {
            this.sendWebSocketMessage({
              type: 'JOIN_FEDERATED_LEARNING',
              participantId: this.participantId
            });
          }
          
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('❌ Disconnected from federated learning server');
          this.isConnected = false;
          this.emit('disconnected', true);
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.emit('error', error);
          resolve(false);
        };

      } catch (error) {
        console.error('Error connecting to WebSocket:', error);
        resolve(false);
      }
    });
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
      
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
      
      setTimeout(() => {
        this.connectWebSocket();
      }, delay);
    } else {
      console.log('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached', true);
    }
  }

  private sendWebSocketMessage(message: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private handleWebSocketMessage(data: any) {
    switch (data.type) {
      case 'INITIAL_DATA':
        this.emit('initialData', data.data);
        break;
      case 'JOINED_FEDERATED_LEARNING':
        this.saveParticipantId(data.data.participantId);
        this.emit('joinedFederatedLearning', data.data);
        break;
      case 'GLOBAL_MODEL_UPDATE':
        this.emit('globalModelUpdate', data.data);
        break;
      case 'PARTICIPANT_UPDATE':
        this.emit('participantUpdate', data.data);
        break;
      case 'MODEL_UPDATE_RECEIVED':
        this.emit('modelUpdateReceived', data.data);
        break;
      case 'FEEDBACK_RECEIVED':
        this.emit('feedbackReceived', data.data);
        break;
    }
  }

  // REST API methods
  async joinFederatedLearning(deviceInfo: any = {}): Promise<{ participantId: string; globalModel: GlobalModel }> {
    try {
      const response = await fetch(`${this.baseUrl}/federated-learning/join`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ deviceInfo }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      await this.saveParticipantId(data.participantId);
      
      // Also join via WebSocket if connected
      if (this.isConnected) {
        this.sendWebSocketMessage({
          type: 'JOIN_FEDERATED_LEARNING',
          participantId: data.participantId,
          deviceInfo
        });
      }

      return data;
    } catch (error) {
      console.error('Error joining federated learning:', error);
      throw error;
    }
  }

  async submitModelUpdate(modelUpdate: Omit<ModelUpdate, 'participantId' | 'timestamp'>): Promise<any> {
    if (!this.participantId) {
      throw new Error('Not joined to federated learning');
    }

    try {
      const updateData = {
        participantId: this.participantId,
        ...modelUpdate,
        timestamp: new Date()
      };

      // Send via WebSocket if connected, otherwise use REST API
      if (this.isConnected) {
        this.sendWebSocketMessage({
          type: 'SUBMIT_MODEL_UPDATE',
          ...updateData
        });
        return { status: 'sent_via_websocket' };
      } else {
        const response = await fetch(`${this.baseUrl}/federated-learning/update`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(updateData),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
      }
    } catch (error) {
      console.error('Error submitting model update:', error);
      throw error;
    }
  }

  async getGlobalModel(): Promise<GlobalModel> {
    try {
      const response = await fetch(`${this.baseUrl}/federated-learning/model`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.globalModel;
    } catch (error) {
      console.error('Error getting global model:', error);
      throw error;
    }
  }

  async submitFeedback(feedback: Omit<FeedbackData, 'participantId'>): Promise<any> {
    if (!this.participantId) {
      throw new Error('Not joined to federated learning');
    }

    try {
      const feedbackData = {
        participantId: this.participantId,
        ...feedback
      };

      // Send via WebSocket if connected, otherwise use REST API
      if (this.isConnected) {
        this.sendWebSocketMessage({
          type: 'SUBMIT_FEEDBACK',
          ...feedbackData
        });
        return { status: 'sent_via_websocket' };
      } else {
        const response = await fetch(`${this.baseUrl}/feedback`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(feedbackData),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      throw error;
    }
  }

  async getFederatedLearningStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/federated-learning/status`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting federated learning status:', error);
      throw error;
    }
  }

  async getAnalytics(timeRange: string = '24h'): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/analytics?timeRange=${timeRange}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting analytics:', error);
      throw error;
    }
  }

  // Simulate local training (in a real app, this would use actual ML training)
  async simulateLocalTraining(trainingData: any[]): Promise<ModelUpdate> {
    return new Promise((resolve) => {
      // Simulate training time
      setTimeout(() => {
        const mockModelWeights = Array.from({ length: 1000 }, () => Math.random() - 0.5);
        
        const update: ModelUpdate = {
          participantId: this.participantId!,
          modelWeights: mockModelWeights,
          trainingMetrics: {
            accuracy: 0.85 + Math.random() * 0.1,
            loss: Math.random() * 0.5,
            epochs: 5,
            dataSize: trainingData.length
          },
          timestamp: new Date()
        };

        resolve(update);
      }, 2000 + Math.random() * 3000); // 2-5 seconds
    });
  }

  // Utility methods
  isParticipant(): boolean {
    return !!this.participantId;
  }

  getParticipantId(): string | null {
    return this.participantId;
  }

  isWebSocketConnected(): boolean {
    return this.isConnected;
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
  }
}

export default new FederatedLearningService();