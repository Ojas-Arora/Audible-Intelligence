import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ScrollView,
  ActivityIndicator,
  Switch,
} from 'react-native';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';

interface InferenceResult {
  event_type: string;
  confidence: number;
  inference_time_ms: number;
  privacy_status: string;
  data_transmitted: boolean;
  timestamp: string;
  audio_duration_ms: number;
  all_predictions: Record<string, number>;
}

interface PrivacyMetrics {
  total_audio_processed_bytes: number;
  events_detected: number;
  privacy_violations: number;
  processing_active: boolean;
  buffer_size_bytes: number;
  privacy_status: string;
  data_transmitted: boolean;
}

interface ModelInfo {
  input_shape: number[];
  input_dtype: string;
  output_shape: number[];
  output_dtype: string;
  model_size_bytes: number;
  inference_count: number;
  avg_inference_time_ms: number;
  quantization_enabled: boolean;
  privacy_status: string;
}

const OnDeviceInference: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [latestResult, setLatestResult] = useState<InferenceResult | null>(null);
  const [results, setResults] = useState<InferenceResult[]>([]);
  const [privacyMetrics, setPrivacyMetrics] = useState<PrivacyMetrics | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [privacyMode, setPrivacyMode] = useState(true);
  const [audioPermission, setAudioPermission] = useState(false);
  
  const recordingRef = useRef<Audio.Recording | null>(null);
  const processingIntervalRef = useRef<NodeJS.Timeout | number | null>(null);

  useEffect(() => {
    initializeAudio();
    initializeModel();
    return () => {
      cleanup();
    };
  }, []);

  const initializeAudio = async () => {
    try {
      const { status } = await Audio.requestPermissionsAsync();
      setAudioPermission(status === 'granted');
      
      if (status === 'granted') {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
          staysActiveInBackground: true,
          interruptionModeIOS: Audio.INTERRUPTION_MODE_IOS_DO_NOT_MIX,
          interruptionModeAndroid: Audio.INTERRUPTION_MODE_ANDROID_DO_NOT_MIX_WITH_OTHERS,
          shouldDuckAndroid: true,
          playThroughEarpieceAndroid: false,
        });
      }
    } catch (error) {
      console.error('Error initializing audio:', error);
      Alert.alert('Error', 'Failed to initialize audio permissions');
    }
  };

  const initializeModel = async () => {
    try {
      setIsProcessing(true);
      
      // In a real implementation, this would load the TensorFlow Lite model
      // For now, we'll simulate the model loading
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setModelLoaded(true);
      setModelInfo({
        input_shape: [1, 128, 128, 1],
        input_dtype: 'float32',
        output_shape: [1, 12],
        output_dtype: 'float32',
        model_size_bytes: 2048576, // ~2MB
        inference_count: 0,
        avg_inference_time_ms: 0,
        quantization_enabled: true,
        privacy_status: 'local_only'
      });
      
      console.log('Model loaded successfully (offline)');
    } catch (error) {
      console.error('Error loading model:', error);
      Alert.alert('Error', 'Failed to load inference model');
    } finally {
      setIsProcessing(false);
    }
  };

  const startRecording = async () => {
    if (!audioPermission) {
      Alert.alert('Permission Required', 'Audio recording permission is required');
      return;
    }

    try {
      setIsRecording(true);
      
      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync({
        android: {
          extension: '.wav',
          outputFormat: Audio.RECORDING_OPTION_ANDROID_OUTPUT_FORMAT_PCM_16BIT,
          audioEncoder: Audio.RECORDING_OPTION_ANDROID_AUDIO_ENCODER_PCM_16BIT,
          sampleRate: 22050,
          numberOfChannels: 1,
          bitRate: 128000,
        },
        ios: {
          extension: '.wav',
          outputFormat: Audio.RECORDING_OPTION_IOS_OUTPUT_FORMAT_LINEARPCM,
          audioQuality: Audio.RECORDING_OPTION_IOS_AUDIO_QUALITY_HIGH,
          sampleRate: 22050,
          numberOfChannels: 1,
          bitRate: 128000,
          linearPCMBitDepth: 16,
          linearPCMIsBigEndian: false,
          linearPCMIsFloat: false,
        },
        web: {
          mimeType: undefined,
          bitsPerSecond: undefined
        }
      });
      
      await recording.startAsync();
      recordingRef.current = recording;
      
      // Start processing loop
      startProcessingLoop();
      
      console.log('Recording started (privacy mode: local processing only)');
    } catch (error) {
      console.error('Error starting recording:', error);
      Alert.alert('Error', 'Failed to start recording');
      setIsRecording(false);
    }
  };

  const stopRecording = async () => {
    try {
      if (recordingRef.current) {
        await recordingRef.current.stopAndUnloadAsync();
        recordingRef.current = null;
      }
      
      setIsRecording(false);
      stopProcessingLoop();
      
      console.log('Recording stopped');
    } catch (error) {
      console.error('Error stopping recording:', error);
    }
  };

  const startProcessingLoop = () => {
    processingIntervalRef.current = setInterval(async () => {
      if (recordingRef.current && modelLoaded) {
        try {
          // Get recording status
          const status = await recordingRef.current.getStatusAsync();
          
          if (status.isRecording) {
            // In a real implementation, this would process audio chunks
            // For now, we'll simulate inference results
            await simulateInference();
          }
        } catch (error) {
          console.error('Error in processing loop:', error);
        }
      }
    }, 1000); // Process every second
  };

  const stopProcessingLoop = () => {
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
  };

  const simulateInference = async () => {
    // Simulate inference result
    const eventTypes = ['dog_bark', 'car_horn', 'alarm', 'glass_break', 'door_slam', 'siren', 'footsteps', 'speech', 'music', 'machinery', 'nature', 'silence'];
    const randomEvent = eventTypes[Math.floor(Math.random() * eventTypes.length)];
    const confidence = 0.6 + Math.random() * 0.4;
    
    const result: InferenceResult = {
      event_type: randomEvent,
      confidence: confidence,
      inference_time_ms: 50 + Math.random() * 100,
      privacy_status: 'local_only',
      data_transmitted: false,
      timestamp: new Date().toISOString(),
      audio_duration_ms: 2000,
      all_predictions: eventTypes.reduce((acc, event) => {
        acc[event] = Math.random() * 0.3;
        return acc;
      }, {} as Record<string, number>)
    };
    
    // Update results
    setLatestResult(result);
    setResults(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 results
    
    // Update metrics
    updatePrivacyMetrics();
  };

  const updatePrivacyMetrics = () => {
    setPrivacyMetrics({
      total_audio_processed_bytes: Math.random() * 1000000,
      events_detected: results.length + 1,
      privacy_violations: 0,
      processing_active: isRecording,
      buffer_size_bytes: 44100 * 4, // 2 seconds of float32 audio
      privacy_status: 'local_only',
      data_transmitted: false
    });
  };

  const cleanup = () => {
    stopRecording();
    stopProcessingLoop();
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getEventIcon = (eventType: string) => {
    const iconMap: Record<string, string> = {
      'dog_bark': 'paw',
      'car_horn': 'car',
      'alarm': 'warning',
      'glass_break': 'alert-circle',
      'door_slam': 'door-open',
      'siren': 'volume-high',
      'footsteps': 'footsteps',
      'speech': 'chatbubble',
      'music': 'musical-notes',
      'machinery': 'construct',
      'nature': 'leaf',
      'silence': 'volume-mute'
    };
    return iconMap[eventType] || 'help-circle';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return '#4CAF50';
    if (confidence > 0.6) return '#FF9800';
    return '#F44336';
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>On-Device Audio Inference</Text>
        <Text style={styles.subtitle}>Privacy-Preserving Local Processing</Text>
      </View>

      {/* Privacy Mode Toggle */}
      <View style={styles.privacySection}>
        <View style={styles.privacyRow}>
          <Ionicons name="shield-checkmark" size={24} color="#4CAF50" />
          <Text style={styles.privacyText}>Privacy Mode</Text>
          <Switch
            value={privacyMode}
            onValueChange={setPrivacyMode}
            trackColor={{ false: '#767577', true: '#4CAF50' }}
            thumbColor={privacyMode ? '#fff' : '#f4f3f4'}
            disabled={isRecording}
          />
        </View>
        <Text style={styles.privacyDescription}>
          All audio processing happens locally on your device. No data is transmitted to external servers.
        </Text>
      </View>

      {/* Model Status */}
      <View style={styles.statusSection}>
        <View style={styles.statusRow}>
          <Text style={styles.statusLabel}>Model Status:</Text>
          <View style={styles.statusIndicator}>
            {modelLoaded ? (
              <>
                <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
                <Text style={[styles.statusText, styles.statusSuccess]}>Loaded (Offline)</Text>
              </>
            ) : (
              <>
                <ActivityIndicator size="small" color="#FF9800" />
                <Text style={[styles.statusText, styles.statusLoading]}>Loading...</Text>
              </>
            )}
          </View>
        </View>
        
        {modelInfo && (
          <View style={styles.modelInfo}>
            <Text style={styles.modelInfoText}>Model Size: {formatBytes(modelInfo.model_size_bytes)}</Text>
            <Text style={styles.modelInfoText}>Quantization: {modelInfo.quantization_enabled ? 'Enabled' : 'Disabled'}</Text>
            <Text style={styles.modelInfoText}>Input Shape: {modelInfo.input_shape.join('x')}</Text>
          </View>
        )}
      </View>

      {/* Recording Controls */}
      <View style={styles.controlsSection}>
        <TouchableOpacity
          style={[styles.recordButton, isRecording && styles.recordButtonActive]}
          onPress={isRecording ? stopRecording : startRecording}
          disabled={!modelLoaded || isProcessing}
        >
          <Ionicons 
            name={isRecording ? "stop" : "mic"} 
            size={32} 
            color={isRecording ? "#fff" : "#fff"} 
          />
          <Text style={styles.recordButtonText}>
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Latest Result */}
      {latestResult && (
        <View style={styles.resultSection}>
          <Text style={styles.sectionTitle}>Latest Detection</Text>
          <View style={styles.resultCard}>
            <View style={styles.resultHeader}>
              <Ionicons 
                name={getEventIcon(latestResult.event_type) as any} 
                size={32} 
                color={getConfidenceColor(latestResult.confidence)} 
              />
              <View style={styles.resultInfo}>
                <Text style={styles.eventType}>{latestResult.event_type.replace('_', ' ').toUpperCase()}</Text>
                <Text style={styles.confidence}>
                  Confidence: {(latestResult.confidence * 100).toFixed(1)}%
                </Text>
                <Text style={styles.inferenceTime}>
                  Inference: {latestResult.inference_time_ms.toFixed(1)}ms
                </Text>
              </View>
            </View>
            <View style={styles.privacyBadge}>
              <Ionicons name="shield-checkmark" size={16} color="#4CAF50" />
              <Text style={styles.privacyBadgeText}>Local Processing</Text>
            </View>
          </View>
        </View>
      )}

      {/* Privacy Metrics */}
      {privacyMetrics && (
        <View style={styles.metricsSection}>
          <Text style={styles.sectionTitle}>Privacy Metrics</Text>
          <View style={styles.metricsGrid}>
            <View style={styles.metricCard}>
              <Text style={styles.metricValue}>{privacyMetrics.events_detected}</Text>
              <Text style={styles.metricLabel}>Events Detected</Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricValue}>{formatBytes(privacyMetrics.total_audio_processed_bytes)}</Text>
              <Text style={styles.metricLabel}>Audio Processed</Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricValue}>{privacyMetrics.privacy_violations}</Text>
              <Text style={styles.metricLabel}>Privacy Violations</Text>
            </View>
            <View style={styles.metricCard}>
              <Text style={styles.metricValue}>{privacyMetrics.data_transmitted ? 'Yes' : 'No'}</Text>
              <Text style={styles.metricLabel}>Data Transmitted</Text>
            </View>
          </View>
        </View>
      )}

      {/* Recent Results */}
      {results.length > 0 && (
        <View style={styles.historySection}>
          <Text style={styles.sectionTitle}>Recent Detections</Text>
          {results.slice(0, 5).map((result, index) => (
            <View key={index} style={styles.historyItem}>
              <Ionicons 
                name={getEventIcon(result.event_type) as any} 
                size={20} 
                color={getConfidenceColor(result.confidence)} 
              />
              <Text style={styles.historyEvent}>{result.event_type.replace('_', ' ')}</Text>
              <Text style={styles.historyConfidence}>
                {(result.confidence * 100).toFixed(0)}%
              </Text>
              <Text style={styles.historyTime}>
                {new Date(result.timestamp).toLocaleTimeString()}
              </Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 5,
  },
  privacySection: {
    margin: 20,
    padding: 15,
    backgroundColor: '#fff',
    borderRadius: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  privacyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  privacyText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
    marginLeft: 10,
  },
  privacyDescription: {
    fontSize: 12,
    color: '#666',
    lineHeight: 16,
  },
  statusSection: {
    margin: 20,
    padding: 15,
    backgroundColor: '#fff',
    borderRadius: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  statusLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusText: {
    marginLeft: 8,
    fontSize: 14,
    fontWeight: '500',
  },
  statusSuccess: {
    color: '#4CAF50',
  },
  statusLoading: {
    color: '#FF9800',
  },
  modelInfo: {
    marginTop: 10,
  },
  modelInfoText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 2,
  },
  controlsSection: {
    alignItems: 'center',
    margin: 20,
  },
  recordButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2196F3',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  recordButtonActive: {
    backgroundColor: '#F44336',
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 10,
  },
  resultSection: {
    margin: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  resultCard: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  resultInfo: {
    marginLeft: 15,
    flex: 1,
  },
  eventType: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  confidence: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  inferenceTime: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  privacyBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E8',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
    alignSelf: 'flex-start',
  },
  privacyBadgeText: {
    fontSize: 12,
    color: '#4CAF50',
    marginLeft: 5,
    fontWeight: '500',
  },
  metricsSection: {
    margin: 20,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricCard: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 10,
    width: '48%',
    marginBottom: 10,
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
    textAlign: 'center',
  },
  historySection: {
    margin: 20,
    marginBottom: 40,
  },
  historyItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  historyEvent: {
    flex: 1,
    fontSize: 14,
    color: '#333',
    marginLeft: 10,
    textTransform: 'capitalize',
  },
  historyConfidence: {
    fontSize: 12,
    color: '#666',
    marginRight: 10,
  },
  historyTime: {
    fontSize: 12,
    color: '#999',
  },
});

export default OnDeviceInference; 