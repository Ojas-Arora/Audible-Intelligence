import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Dimensions,
  Platform,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Mic, MicOff, Zap, Shield, Volume2, TriangleAlert as AlertTriangle, Brain, Activity } from 'lucide-react-native';
import PyTorchModelInterface from '@/components/PyTorchModelInterface';
import { useTheme } from '@/components/ThemeProvider';
import { useAppSettings } from '@/hooks/useAppSettings';
import { useLiveEvents } from '@/hooks/useLiveEvents';

const { width, height } = Dimensions.get('window');

interface DetectedEvent {
  type: string;
  confidence: number;
  timestamp: Date;
  icon: string;
  category: string;
  processing_time_ms?: number;
}

export default function DetectionScreen() {
  const { theme, isDark } = useTheme();
  const { settings } = useAppSettings();
  const { addEvent } = useLiveEvents();
  const [isRecording, setIsRecording] = useState(false);
  const [currentEvent, setCurrentEvent] = useState<DetectedEvent | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recentEvents, setRecentEvents] = useState<DetectedEvent[]>([]);
  const [isHovered, setIsHovered] = useState(false);
  
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const waveAnim = useRef(new Animated.Value(0)).current;
  const confidenceAnim = useRef(new Animated.Value(0)).current;

  // Event type mapping with icons and colors
  const eventTypeMapping = {
    'dog_bark': { icon: '🐕', color: theme.colors.dogBark },
    'car_horn': { icon: '🚗', color: theme.colors.carHorn },
    'alarm': { icon: '🚨', color: theme.colors.alarm },
    'glass_break': { icon: '🥃', color: theme.colors.glassBreak },
    'siren': { icon: '🚑', color: theme.colors.siren },
    'door_slam': { icon: '🚪', color: theme.colors.doorSlam },
    'footsteps': { icon: '👣', color: theme.colors.footsteps },
    'speech': { icon: '🗣️', color: theme.colors.speech },
    'music': { icon: '🎵', color: theme.colors.music },
    'machinery': { icon: '⚙️', color: theme.colors.machinery },
    'nature': { icon: '🌿', color: theme.colors.nature },
    'silence': { icon: '🔇', color: theme.colors.silence },
  };

  useEffect(() => {
    // Auto-start detection if enabled in settings
    if (settings.autoDetection) {
      setIsRecording(true);
    }
  }, [settings.autoDetection]);

  useEffect(() => {
    if (isRecording) {
      startPulseAnimation();
      startWaveAnimation();
      simulateAudioLevel();
    } else {
      stopAnimations();
    }
  }, [isRecording]);

  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const startWaveAnimation = () => {
    Animated.loop(
      Animated.timing(waveAnim, {
        toValue: 1,
        duration: 2000,
        useNativeDriver: true,
      })
    ).start();
  };

  const stopAnimations = () => {
    pulseAnim.stopAnimation();
    waveAnim.stopAnimation();
    confidenceAnim.stopAnimation();
  };

  const simulateAudioLevel = () => {
    const interval = setInterval(() => {
      if (!isRecording) {
        clearInterval(interval);
        return;
      }
      setAudioLevel(20 + Math.random() * 60);
    }, 100);
  };

  const handlePyTorchPrediction = (prediction: any) => {
    // Apply confidence threshold from settings
    if (prediction.confidence < settings.confidenceThreshold) {
      return;
    }

    const eventMapping = eventTypeMapping[prediction.event_type as keyof typeof eventTypeMapping];
    
    if (eventMapping) {
      const event: DetectedEvent = {
        type: prediction.event_type,
        confidence: prediction.confidence,
        timestamp: new Date(),
        icon: eventMapping.icon,
        category: prediction.category,
        processing_time_ms: prediction.processing_time_ms,
      };

      setCurrentEvent(event);
      
      if (settings.saveDetections) {
        setRecentEvents(prev => [event, ...prev.slice(0, 4)]);
      }
      
      // Add to analytics
      addEvent(event);
      
      Animated.timing(confidenceAnim, {
        toValue: prediction.confidence,
        duration: 500,
        useNativeDriver: false,
      }).start();

      // Haptic feedback if enabled and not on web
      if (settings.hapticFeedback && Platform.OS !== 'web') {
        // Note: Haptics would be implemented here for native platforms
      }

      setTimeout(() => {
        setCurrentEvent(null);
        confidenceAnim.setValue(0);
      }, 4000);
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    setCurrentEvent(null);
    setAudioLevel(0);
    
    if (!isRecording && !settings.saveDetections) {
      setRecentEvents([]);
    }
  };

  const getWaveTransform = () => {
    const translateY = waveAnim.interpolate({
      inputRange: [0, 1],
      outputRange: [0, -20],
    });
    return { transform: [{ translateY }] };
  };

  const getEventColor = (eventType: string) => {
    const mapping = eventTypeMapping[eventType as keyof typeof eventTypeMapping];
    return mapping?.color || theme.colors.textSecondary;
  };

  const renderContent = () => (
    <ScrollView contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={[styles.title, { color: theme.colors.primary }]}>Live Detection</Text>
        <Text style={[styles.subtitle, { color: theme.colors.textSecondary }]}>
          Real-time PyTorch • Privacy First • Edge Computing
        </Text>
      </View>

      {/* Live Detection Status */}
      <View style={[styles.statusCard, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}>
        <View style={styles.statusIndicator}>
          <View style={[styles.statusDot, { backgroundColor: isRecording ? theme.colors.success : theme.colors.error }]} />
          <Text style={[styles.statusText, { color: theme.colors.text }]}>
            {isRecording ? 'Listening for events...' : 'Detection Paused'}
          </Text>
        </View>
        <TouchableOpacity
          style={[styles.toggleButton, { backgroundColor: isRecording ? theme.colors.error : theme.colors.primary }]}
          onPress={toggleRecording}
        >
          {isRecording ? <MicOff size={18} color="white" /> : <Mic size={18} color="white" />}
          <Text style={styles.toggleButtonText}>{isRecording ? 'Stop' : 'Start'} Listening</Text>
        </TouchableOpacity>
      </View>

      {/* Animated Waveform Placeholder */}
      <View style={styles.waveformContainer}>
        <Animated.View style={[
          styles.waveform,
          {
            backgroundColor: isDark ? 'rgba(0,255,208,0.18)' : 'rgba(0, 119, 182, 0.1)',
            borderColor: isDark ? '#00ffd0' : theme.colors.primary,
            shadowColor: isDark ? '#00ffd0' : theme.colors.primary,
            opacity: isRecording ? 1 : 0.5,
            transform: [{ scaleY: 1 + audioLevel / 100 }],
          }
        ]} />
        <Text style={[styles.waveformLabel, { color: isDark ? '#00ffd0' : theme.colors.primary }]}>Audio Waveform</Text>
      </View>

      {/* Current Detected Event */}
      {currentEvent && (
        <View style={[
          styles.currentEventContainer,
          {
            borderColor: theme.colors.accent,
            backgroundColor: isDark ? 'rgba(162,89,255,0.13)' : theme.colors.accent + '20',
            shadowColor: theme.colors.accent,
          }
        ]}>
          <Text style={[styles.currentEventTitle, { color: theme.colors.accent }]}>Detected Event</Text>
          <View style={styles.currentEventRow}>
            <Text style={styles.currentEventIcon}>{currentEvent.icon}</Text>
            <Text style={[styles.currentEventType, { color: getEventColor(currentEvent.type) }]}>{currentEvent.type.replace('_', ' ').toUpperCase()}</Text>
            <View style={[styles.confidenceBadge, { backgroundColor: theme.colors.primary + '22' }]}>
              <Text style={[styles.confidenceText, { color: theme.colors.primary }]}>{Math.round(currentEvent.confidence * 100)}%</Text>
            </View>
          </View>
          {currentEvent.processing_time_ms && (
            <Text style={[styles.processingTime, { color: theme.colors.textSecondary }]}>Processing: {currentEvent.processing_time_ms.toFixed(1)} ms</Text>
          )}
        </View>
      )}

      {/* Recent Events Feed */}
      <View
        style={[styles.recentEventsContainer, {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.colors.primary,
        }]}
      >
        <Text style={[styles.recentEventsTitle, { color: theme.colors.primary }]}>Recent Events</Text>
        {recentEvents.length === 0 ? (
          <Text style={[styles.emptyFeedText, { color: theme.colors.textSecondary }]}>No events detected yet.</Text>
        ) : (
          recentEvents.map((event, idx) => (
            <View key={idx} style={[styles.recentEventItem, { borderBottomColor: theme.colors.border }]}>
              <Text style={styles.recentEventIcon}>{event.icon}</Text>
              <Text style={[styles.recentEventType, { color: getEventColor(event.type) }]}>{event.type.replace('_', ' ').toUpperCase()}</Text>
              <Text style={[styles.recentEventConfidence, { color: theme.colors.success }]}>{Math.round(event.confidence * 100)}%</Text>
              <Text style={[styles.recentEventTime, { color: theme.colors.textSecondary }]}>{formatTimeAgo(event.timestamp)}</Text>
            </View>
          ))
        )}
      </View>
    </ScrollView>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: isDark ? '#0f1123' : theme.colors.background }]}>
      {isDark ? (
        <LinearGradient colors={theme.gradients.background} style={styles.container}>
          {renderContent()}
        </LinearGradient>
      ) : (
        renderContent()
      )}
    </SafeAreaView>
  );
}

function formatTimeAgo(date: Date) {
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return date.toLocaleDateString();
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 24,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 16,
    marginTop: 8,
    textAlign: 'center',
  },
  statusCard: {
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
    marginBottom: 24,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
  },
  toggleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  toggleButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
    marginLeft: 8,
  },
  waveformContainer: {
    marginTop: 28,
    alignItems: 'center',
    justifyContent: 'center',
    height: 80,
  },
  waveform: {
    width: width * 0.7,
    height: 40,
    borderRadius: 20,
    marginBottom: 8,
    shadowOpacity: 0.85,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 6 },
    borderWidth: 2,
  },
  waveformLabel: {
    fontSize: 13,
    fontStyle: 'italic',
  },
  currentEventContainer: {
    marginTop: 24,
    padding: 24,
    borderRadius: 28,
    borderWidth: 2.5,
    shadowOpacity: 0.7,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 8 },
    alignItems: 'center',
    marginBottom: 24,
    elevation: 8,
  },
  currentEventTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  currentEventRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  currentEventIcon: {
    fontSize: 32,
    marginRight: 10,
  },
  currentEventType: {
    fontSize: 18,
    fontWeight: 'bold',
    marginRight: 10,
  },
  confidenceBadge: {
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
    marginLeft: 4,
  },
  confidenceText: {
    color: '#aeefff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  processingTime: {
    color: '#aeefff',
    fontSize: 15,
  },
  recentEventsContainer: {
    borderRadius: 28,
    borderWidth: 2.5,
    shadowOpacity: 0.18,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: 10 },
    padding: 28,
    marginBottom: 28,
    elevation: 12,
  },
  recentEventsTitle: {
    fontWeight: 'bold',
    fontSize: 20,
    marginBottom: 10,
  },
  recentEventItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  recentEventIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  recentEventType: {
    fontWeight: 'bold',
    fontSize: 16,
    marginRight: 10,
  },
  recentEventConfidence: {
    fontWeight: 'bold',
    fontSize: 16,
    marginRight: 10,
  },
  recentEventTime: {
    fontSize: 14,
  },
  emptyFeedText: {
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 20,
  },
});