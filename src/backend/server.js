const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const http = require('http');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

// In-memory storage (in production, use a proper database)
let federatedLearningData = {
  globalModel: {
    version: '2.1.0',
    accuracy: 0.942,
    participants: 12547,
    lastUpdate: new Date(),
    modelWeights: null
  },
  participants: new Map(),
  trainingRounds: [],
  aggregatedUpdates: []
};

let connectedClients = new Set();

// WebSocket connection handling
wss.on('connection', (ws) => {
  console.log('New client connected');
  connectedClients.add(ws);

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      handleWebSocketMessage(ws, data);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
    connectedClients.delete(ws);
  });

  // Send initial data
  ws.send(JSON.stringify({
    type: 'INITIAL_DATA',
    data: {
      globalModel: federatedLearningData.globalModel,
      participantCount: federatedLearningData.participants.size
    }
  }));
});

function handleWebSocketMessage(ws, data) {
  switch (data.type) {
    case 'JOIN_FEDERATED_LEARNING':
      handleJoinFederatedLearning(ws, data);
      break;
    case 'SUBMIT_MODEL_UPDATE':
      handleModelUpdate(ws, data);
      break;
    case 'REQUEST_GLOBAL_MODEL':
      handleGlobalModelRequest(ws, data);
      break;
    case 'SUBMIT_FEEDBACK':
      handleFeedbackSubmission(ws, data);
      break;
  }
}

function handleJoinFederatedLearning(ws, data) {
  const participantId = data.participantId || generateParticipantId();
  
  federatedLearningData.participants.set(participantId, {
    id: participantId,
    joinedAt: new Date(),
    contributions: 0,
    lastActive: new Date(),
    deviceInfo: data.deviceInfo || {}
  });

  ws.participantId = participantId;

  // Broadcast updated participant count
  broadcastToAll({
    type: 'PARTICIPANT_UPDATE',
    data: {
      participantCount: federatedLearningData.participants.size,
      newParticipant: participantId
    }
  });

  ws.send(JSON.stringify({
    type: 'JOINED_FEDERATED_LEARNING',
    data: {
      participantId,
      globalModel: federatedLearningData.globalModel
    }
  }));
}

function handleModelUpdate(ws, data) {
  const participantId = ws.participantId;
  if (!participantId) return;

  const participant = federatedLearningData.participants.get(participantId);
  if (!participant) return;

  // Store the model update
  const update = {
    participantId,
    timestamp: new Date(),
    modelWeights: data.modelWeights,
    trainingMetrics: data.trainingMetrics,
    dataSize: data.dataSize || 0
  };

  federatedLearningData.aggregatedUpdates.push(update);
  participant.contributions++;
  participant.lastActive = new Date();

  // Simulate federated averaging (in production, use proper FL algorithms)
  if (federatedLearningData.aggregatedUpdates.length >= 5) {
    performFederatedAveraging();
  }

  ws.send(JSON.stringify({
    type: 'MODEL_UPDATE_RECEIVED',
    data: {
      updateId: update.timestamp.getTime(),
      status: 'accepted',
      contributionCount: participant.contributions
    }
  }));
}

function performFederatedAveraging() {
  // Simulate federated averaging algorithm
  const updates = federatedLearningData.aggregatedUpdates.slice(-10);
  
  // Update global model
  federatedLearningData.globalModel.version = incrementVersion(federatedLearningData.globalModel.version);
  federatedLearningData.globalModel.accuracy = Math.min(0.99, federatedLearningData.globalModel.accuracy + Math.random() * 0.01);
  federatedLearningData.globalModel.lastUpdate = new Date();

  // Create training round record
  const trainingRound = {
    id: federatedLearningData.trainingRounds.length + 1,
    timestamp: new Date(),
    participantCount: updates.length,
    accuracyImprovement: Math.random() * 0.02,
    modelVersion: federatedLearningData.globalModel.version
  };

  federatedLearningData.trainingRounds.push(trainingRound);

  // Clear processed updates
  federatedLearningData.aggregatedUpdates = [];

  // Broadcast new global model to all participants
  broadcastToAll({
    type: 'GLOBAL_MODEL_UPDATE',
    data: {
      globalModel: federatedLearningData.globalModel,
      trainingRound
    }
  });

  console.log(`Federated averaging completed. New model version: ${federatedLearningData.globalModel.version}`);
}

function handleGlobalModelRequest(ws, data) {
  ws.send(JSON.stringify({
    type: 'GLOBAL_MODEL_RESPONSE',
    data: {
      globalModel: federatedLearningData.globalModel,
      downloadUrl: '/api/model/download'
    }
  }));
}

function handleFeedbackSubmission(ws, data) {
  const participantId = ws.participantId;
  if (!participantId) return;

  // Store feedback for model improvement
  const feedback = {
    participantId,
    timestamp: new Date(),
    predictedEvent: data.predictedEvent,
    actualEvent: data.actualEvent,
    confidence: data.confidence,
    isCorrect: data.predictedEvent === data.actualEvent
  };

  // In production, store in database and use for model retraining
  console.log('Received feedback:', feedback);

  ws.send(JSON.stringify({
    type: 'FEEDBACK_RECEIVED',
    data: {
      feedbackId: feedback.timestamp.getTime(),
      status: 'processed'
    }
  }));
}

function broadcastToAll(message) {
  connectedClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(message));
    }
  });
}

function generateParticipantId() {
  return 'participant_' + Math.random().toString(36).substr(2, 9);
}

function incrementVersion(version) {
  const parts = version.split('.');
  parts[2] = (parseInt(parts[2]) + 1).toString();
  return parts.join('.');
}

// REST API Endpoints

// Get federated learning status
app.get('/api/federated-learning/status', (req, res) => {
  res.json({
    globalModel: federatedLearningData.globalModel,
    participantCount: federatedLearningData.participants.size,
    trainingRounds: federatedLearningData.trainingRounds.slice(-10),
    isActive: true
  });
});

// Join federated learning
app.post('/api/federated-learning/join', (req, res) => {
  const participantId = generateParticipantId();
  
  federatedLearningData.participants.set(participantId, {
    id: participantId,
    joinedAt: new Date(),
    contributions: 0,
    lastActive: new Date(),
    deviceInfo: req.body.deviceInfo || {}
  });

  res.json({
    participantId,
    globalModel: federatedLearningData.globalModel,
    status: 'joined'
  });
});

// Submit model update
app.post('/api/federated-learning/update', (req, res) => {
  const { participantId, modelWeights, trainingMetrics, dataSize } = req.body;
  
  const participant = federatedLearningData.participants.get(participantId);
  if (!participant) {
    return res.status(404).json({ error: 'Participant not found' });
  }

  const update = {
    participantId,
    timestamp: new Date(),
    modelWeights,
    trainingMetrics,
    dataSize: dataSize || 0
  };

  federatedLearningData.aggregatedUpdates.push(update);
  participant.contributions++;
  participant.lastActive = new Date();

  // Trigger federated averaging if enough updates
  if (federatedLearningData.aggregatedUpdates.length >= 5) {
    performFederatedAveraging();
  }

  res.json({
    updateId: update.timestamp.getTime(),
    status: 'accepted',
    contributionCount: participant.contributions,
    nextAggregation: federatedLearningData.aggregatedUpdates.length >= 5 ? 'now' : `${5 - federatedLearningData.aggregatedUpdates.length} updates remaining`
  });
});

// Get global model
app.get('/api/federated-learning/model', (req, res) => {
  res.json({
    globalModel: federatedLearningData.globalModel,
    downloadUrl: '/api/model/download'
  });
});

// Download model file (simulated)
app.get('/api/model/download', (req, res) => {
  // In production, serve actual model file
  const modelData = {
    version: federatedLearningData.globalModel.version,
    weights: 'base64_encoded_model_weights_here',
    metadata: {
      accuracy: federatedLearningData.globalModel.accuracy,
      classes: ['dog_bark', 'car_horn', 'alarm', 'glass_break', 'door_slam', 'siren', 'footsteps', 'speech', 'music', 'machinery', 'nature', 'silence'],
      inputShape: [1, 128, 128, 1],
      framework: 'pytorch'
    }
  };

  res.setHeader('Content-Type', 'application/octet-stream');
  res.setHeader('Content-Disposition', `attachment; filename=model_${federatedLearningData.globalModel.version}.json`);
  res.send(JSON.stringify(modelData));
});

// Submit feedback
app.post('/api/feedback', (req, res) => {
  const { participantId, predictedEvent, actualEvent, confidence } = req.body;
  
  const feedback = {
    participantId,
    timestamp: new Date(),
    predictedEvent,
    actualEvent,
    confidence,
    isCorrect: predictedEvent === actualEvent
  };

  // Store feedback (in production, use database)
  console.log('Received feedback via REST:', feedback);

  res.json({
    feedbackId: feedback.timestamp.getTime(),
    status: 'processed',
    message: 'Thank you for your feedback!'
  });
});

// Get analytics data
app.get('/api/analytics', (req, res) => {
  const { timeRange = '24h' } = req.query;
  
  // Generate mock analytics data
  const analytics = {
    totalEvents: Math.floor(Math.random() * 10000) + 5000,
    averageAccuracy: federatedLearningData.globalModel.accuracy,
    participantCount: federatedLearningData.participants.size,
    modelUpdates: federatedLearningData.trainingRounds.length,
    timeRange,
    eventDistribution: {
      'dog_bark': Math.floor(Math.random() * 1000) + 500,
      'car_horn': Math.floor(Math.random() * 800) + 300,
      'alarm': Math.floor(Math.random() * 600) + 200,
      'speech': Math.floor(Math.random() * 1200) + 800,
      'music': Math.floor(Math.random() * 900) + 400,
      'footsteps': Math.floor(Math.random() * 700) + 350
    },
    hourlyData: Array.from({ length: 24 }, (_, i) => ({
      hour: i,
      events: Math.floor(Math.random() * 100) + 20,
      accuracy: 0.85 + Math.random() * 0.1
    }))
  };

  res.json(analytics);
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date(),
    uptime: process.uptime(),
    federatedLearning: {
      active: true,
      participants: federatedLearningData.participants.size,
      modelVersion: federatedLearningData.globalModel.version
    }
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: 'The requested resource was not found'
  });
});

const PORT = process.env.PORT || 3001;

server.listen(PORT, () => {
  console.log(`🚀 AudioSense Backend Server running on port ${PORT}`);
  console.log(`📊 Federated Learning API available at http://localhost:${PORT}/api`);
  console.log(`🔌 WebSocket server running for real-time updates`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});