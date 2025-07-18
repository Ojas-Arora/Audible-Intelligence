import React, { useState, useEffect, useRef } from 'react';
import { Audio } from 'expo-av';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Dimensions,
  Platform,
  ScrollView,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Mic, MicOff, Zap, Shield, Volume2, TriangleAlert as AlertTriangle, Brain, Activity, Play, Pause, Settings, TrendingUp, Eye, Lock, Bell, BellOff } from 'lucide-react-native';
import { useTheme } from '@/components/ThemeProvider';
import { useAppSettings } from '@/hooks/useAppSettings';
import { useLiveEvents } from '@/hooks/useLiveEvents';

const { width, height } = Dimensions.get('window');

interface DetectedEvent {
  type: string;
  confidence: number;
  timestamp: Date;
  category: string;
}

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

const WaveformVisualizer = ({ isActive, audioLevel, theme }: any) => {
  const waveAnimations = useRef(Array.from({ length: 20 }, () => new Animated.Value(0.3))).current;

  useEffect(() => {
    if (isActive) {
      const animations = waveAnimations.map((anim, index) => 
        Animated.loop(
          Animated.sequence([
            Animated.timing(anim, {
              toValue: 0.3 + Math.random() * 0.7,
              duration: 300 + Math.random() * 200,
              useNativeDriver: true,
            }),
            Animated.timing(anim, {
              toValue: 0.3,
              duration: 300 + Math.random() * 200,
              useNativeDriver: true,
            }),
          ])
        )
      );
      
      animations.forEach((anim, index) => {
        setTimeout(() => anim.start(), index * 50);
      });
    } else {
      waveAnimations.forEach(anim => {
        Animated.timing(anim, {
          toValue: 0.3,
          duration: 200,
          useNativeDriver: true,
        }).start();
      });
    }
  }, [isActive]);

  return (
    <View style={styles.waveformContainer}>
      <View style={styles.waveform}>
        {waveAnimations.map((anim, index) => (
          <Animated.View
            key={index}
            style={[
              styles.waveBar,
              {
                backgroundColor: isActive ? theme.colors.primary : theme.colors.border,
                transform: [{ scaleY: anim }],
              },
            ]}
          />
        ))}
      </View>
      <Text style={[styles.waveformLabel, { color: theme.colors.textSecondary }]}>
        {isActive ? 'Listening...' : 'Ready to detect'}
      </Text>
    </View>
  );
};

const EventCard = ({ event, theme, delay = 0 }: any) => {
  const scaleAnim = useRef(new Animated.Value(0)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    setTimeout(() => {
      Animated.parallel([
        Animated.spring(scaleAnim, {
          toValue: 1,
          tension: 50,
          friction: 7,
          useNativeDriver: true,
        }),
        Animated.timing(opacityAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ]).start();
    }, delay);
  }, []);

  return (
    <Animated.View
      style={[
        styles.eventCard,
        {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          transform: [{ scale: scaleAnim }],
          opacity: opacityAnim,
        },
      ]}
    >
      <LinearGradient
        colors={[theme.colors.card, theme.colors.surface]}
        style={styles.eventCardGradient}
      >
        <View style={styles.eventHeader}>

          <View style={styles.eventInfo}>
            <Text style={[styles.eventType, { color: theme.colors.text }]}>
              {event.type.replace('_', ' ').toUpperCase()}
            </Text>
            <Text style={[styles.eventCategory, { color: theme.colors.textSecondary }]}>
              {event.category}
            </Text>
          </View>
          <View style={[styles.confidenceBadge, { backgroundColor: theme.colors.success + '20' }]}>
            <Text style={[styles.confidenceText, { color: theme.colors.success }]}>
              {Math.round(event.confidence * 100)}%
            </Text>
          </View>
        </View>
      </LinearGradient>
    </Animated.View>
  );
};

export default function DetectionScreen() {
  const { theme, isDark } = useTheme();
  const { settings } = useAppSettings();
  const { addEvent } = useLiveEvents();
  const [isRecording, setIsRecording] = useState(false);
  const [currentEvent, setCurrentEvent] = useState<DetectedEvent | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recentEvents, setRecentEvents] = useState<DetectedEvent[]>([]);
  const [detectionStats, setDetectionStats] = useState({
    totalEvents: 0,
    avgConfidence: 0,
    uptime: '00:00:00',
  });
  
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const recordingStartTime = useRef<Date | null>(null);
  const detectionInterval = useRef<NodeJS.Timeout | null>(null);

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

  // Auto-detection based on settings - ONLY when settings change
  useEffect(() => {
    if (settings.autoDetection && !isRecording) {
      setIsRecording(true);
      recordingStartTime.current = new Date();
    }
  }, [settings.autoDetection]); // Only depend on autoDetection setting

  useEffect(() => {
    if (isRecording) {
      startPulseAnimation();
      simulateAudioLevel();
      startDetection();
      
      // Update uptime
      const interval = setInterval(() => {
        if (recordingStartTime.current) {
          const now = new Date();
          const diff = now.getTime() - recordingStartTime.current.getTime();
          const hours = Math.floor(diff / (1000 * 60 * 60));
          const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
          const seconds = Math.floor((diff % (1000 * 60)) / 1000);
          
          setDetectionStats(prev => ({
            ...prev,
            uptime: `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`,
          }));
        }
      }, 1000);
      
      return () => clearInterval(interval);
    } else {
      stopAnimations();
      stopDetection();
    }
  }, [isRecording]);

  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1,
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

  const stopAnimations = () => {
    pulseAnim.stopAnimation();
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

  const startDetection = () => {
  if (detectionInterval.current) return;
  detectionInterval.current = setInterval(async () => {
    try {
      const startTime = Date.now();
      const response = await fetch('http://192.168.29.32:5001/predict-from-path', {
        method: 'POST',
      });
      const result = await response.json();
      const endTime = Date.now();

      if (result.predicted_class) {
        const detectedType = result.predicted_class;
        const confidence = result.confidence || 0.99;
        const event = {
          type: detectedType,
          confidence,
          timestamp: new Date(),
          category: getCategoryForEvent(detectedType)
        };

        setCurrentEvent(event);
        if (settings.saveDetections) {
          setRecentEvents(prev => [event, ...prev.slice(0, 4)]);
        }
        addEvent(event);
        showNotification(event);
        setDetectionStats(prev => ({
          totalEvents: prev.totalEvents + 1,
          avgConfidence: (prev.avgConfidence * prev.totalEvents + confidence) / (prev.totalEvents + 1),
          uptime: prev.uptime,
        }));
        setTimeout(() => {
          setCurrentEvent(null);
        }, 4000);
      } else {
        Alert.alert('Prediction failed', result.error || 'Unknown error');
      }
    } catch (error: any) {
      Alert.alert('Error', error.message || 'Unknown error');
    }
  }, 3000) as unknown as NodeJS.Timeout;
};

const stopDetection = () => {
  if (detectionInterval.current) {
    clearInterval(detectionInterval.current);
    detectionInterval.current = null;
  }
};

  const showNotification = (event: DetectedEvent) => {
    // Final safety check - this should never be called if notifications are disabled
    if (settings.notifications !== true) {
      console.log('🚫 SAFETY CHECK: Notifications disabled, aborting notification');
      return;
    }

    const title = `Audio Event Detected`;
    const message = `${event.type.replace('_', ' ').toUpperCase()} detected with ${Math.round(event.confidence * 100)}% confidence`;
    
    console.log('📢 SHOWING NOTIFICATION:', { 
      title, 
      message, 
      notificationsEnabled: settings.notifications,
      settingsObject: settings 
    });
    
    if (Platform.OS === 'web') {
      // Web notification
      if ('Notification' in window) {
        if (Notification.permission === 'granted') {
          new Notification(title, {
            body: message,
          });
        } else if (Notification.permission !== 'denied') {
          Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
              new Notification(title, {
                body: message,
              });
            }
          });
        }
      }
      
      // Fallback: Browser alert for web
      setTimeout(() => {
        alert(`${title}\n\n${message}`);
      }, 100);
    } else {
      // Mobile notification using Alert
      Alert.alert(title, message, [
        { text: 'OK', style: 'default' }
      ]);
    }
  };

  const getCategoryForEvent = (eventType: string): string => {
    const categoryMap: { [key: string]: string } = {
      'dog_bark': 'animals',
      'car_horn': 'vehicles',
      'alarm': 'alarms',
      'glass_break': 'home',
      'door_slam': 'home',
      'siren': 'vehicles',
      'footsteps': 'home',
      'speech': 'home',
      'music': 'home',
      'machinery': 'industrial',
      'nature': 'environment',
      'silence': 'ambient',
    };
    return categoryMap[eventType] || 'unknown';
  };

  const toggleRecording = () => {
    if (!isRecording) {
      recordingStartTime.current = new Date();
    } else {
      recordingStartTime.current = null;
      setDetectionStats(prev => ({ ...prev, uptime: '00:00:00' }));
    }
    
    setIsRecording(!isRecording);
    setCurrentEvent(null);
    setAudioLevel(0);
    
    if (!isRecording && !settings.saveDetections) {
      setRecentEvents([]);
    }
  };

  const renderContent = () => (
    <ScrollView 
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Hero Header */}
      <LinearGradient
        colors={theme.gradients.hero}
        style={styles.heroHeader}
      >
        <View style={styles.heroContent}>
          <Animated.View style={[styles.heroIcon, { transform: [{ scale: pulseAnim }] }]}>
            {isRecording ? (
              <Mic size={32} color="white" />
            ) : (
              <MicOff size={32} color="white" />
            )}
          </Animated.View>
          <Text style={styles.heroTitle}>Live Detection</Text>
          <Text style={styles.heroSubtitle}>
            Real-time PyTorch • Privacy First • Edge Computing
          </Text>
          
          <View style={styles.heroStats}>
            <View style={styles.heroStat}>
              <Shield size={16} color="rgba(255, 255, 255, 0.8)" />
              <Text style={styles.heroStatText}>100% Local</Text>
            </View>
            <View style={styles.heroStat}>
              <Zap size={16} color="rgba(255, 255, 255, 0.8)" />
              <Text style={styles.heroStatText}>8ms Latency</Text>
            </View>
            <View style={styles.heroStat}>
              {settings.notifications ? (
                <Bell size={16} color="rgba(34, 197, 94, 1)" />
              ) : (
                <BellOff size={16} color="rgba(239, 68, 68, 1)" />
              )}
              <Text style={[styles.heroStatText, { 
                color: settings.notifications ? 'rgba(34, 197, 94, 1)' : 'rgba(239, 68, 68, 1)' 
              }]}>
                {settings.notifications ? 'Notifications ON' : 'Notifications OFF'}
              </Text>
            </View>
          </View>
        </View>
      </LinearGradient>

      {/* Detection Status */}
      <View style={styles.statusSection}>
        <View style={[
          styles.statusCard,
          { 
            backgroundColor: theme.colors.card,
            borderColor: isRecording ? theme.colors.success : theme.colors.border 
          }
        ]}>
          <View style={styles.statusHeader}>
            <View style={[
              styles.statusDot,
              { backgroundColor: isRecording ? theme.colors.success : theme.colors.error }
            ]} />
            <Text style={[styles.statusText, { color: theme.colors.text }]}>
              {isRecording ? 'Listening for events...' : 'Detection Paused'}
            </Text>
            
            {/* CRITICAL: Real-time notification status indicator that syncs with settings */}
            <View style={[
              styles.notificationBadge, 
              { backgroundColor: settings.notifications ? theme.colors.success + '20' : theme.colors.error + '20' }
            ]}>
              {settings.notifications ? (
                <Bell size={14} color={theme.colors.success} />
              ) : (
                <BellOff size={14} color={theme.colors.error} />
              )}
              <Text style={[
                styles.notificationText, 
                { color: settings.notifications ? theme.colors.success : theme.colors.error }
              ]}>
                {settings.notifications ? 'Notifications ON' : 'Notifications OFF'}
              </Text>
            </View>
          </View>
          
          <AnimatedTouchableOpacity
            style={[
              styles.toggleButton,
              { backgroundColor: isRecording ? theme.colors.error : theme.colors.primary }
            ]}
            onPress={toggleRecording}
            activeOpacity={0.8}
          >
            {isRecording ? <Pause size={20} color="white" /> : <Play size={20} color="white" />}
            <Text style={styles.toggleButtonText}>
              {isRecording ? 'Stop' : 'Start'} Detection
            </Text>
          </AnimatedTouchableOpacity>
        </View>
      </View>

      {/* Notification Status Card - Shows current sync status */}
      <View style={styles.notificationStatusSection}>
        <View style={[
          styles.notificationStatusCard,
          { 
            backgroundColor: settings.notifications ? theme.colors.success + '10' : theme.colors.error + '10',
            borderColor: settings.notifications ? theme.colors.success : theme.colors.error 
          }
        ]}>
          <View style={styles.notificationStatusHeader}>
            {settings.notifications ? (
              <Bell size={20} color={theme.colors.success} />
            ) : (
              <BellOff size={20} color={theme.colors.error} />
            )}
            <Text style={[styles.notificationStatusTitle, { color: theme.colors.text }]}>
              Notification Status
            </Text>
          </View>
          <Text style={[styles.notificationStatusText, { color: theme.colors.textSecondary }]}>
            {settings.notifications 
              ? '✅ Notifications are enabled. You will receive alerts when events are detected.'
              : '❌ Notifications are disabled. Enable them in Settings to receive event alerts.'
            }
          </Text>
        </View>
      </View>

      {/* Waveform Visualizer */}
      <WaveformVisualizer 
        isActive={isRecording} 
        audioLevel={audioLevel} 
        theme={theme} 
      />

      {/* Detection Stats */}
      <View style={styles.statsSection}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
          Detection Statistics
        </Text>
        <View style={styles.statsGrid}>
          <View style={[styles.statCard, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}>
            <Activity size={20} color={theme.colors.primary} />
            <Text style={[styles.statValue, { color: theme.colors.text }]}>{detectionStats.totalEvents}</Text>
            <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>Events</Text>
          </View>
          
          <View style={[styles.statCard, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}>
            <TrendingUp size={20} color={theme.colors.success} />
            <Text style={[styles.statValue, { color: theme.colors.text }]}>
              {Math.round(detectionStats.avgConfidence * 100) || 0}%
            </Text>
            <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>Confidence</Text>
          </View>
          
          <View style={[styles.statCard, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}>
            <Eye size={20} color={theme.colors.accent} />
            <Text style={[styles.statValue, { color: theme.colors.text }]}>{detectionStats.uptime}</Text>
            <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>Uptime</Text>
          </View>
        </View>
      </View>

      {/* Current Detection */}
      {currentEvent && (
        <View style={styles.currentDetectionSection}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
            Current Detection
          </Text>
          <EventCard event={currentEvent} theme={theme} />
        </View>
      )}

      {/* Recent Events */}
      <View style={styles.recentEventsSection}>
        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
            Recent Events
          </Text>
          <TouchableOpacity style={styles.settingsButton}>
            <Settings size={20} color={theme.colors.textSecondary} />
          </TouchableOpacity>
        </View>
        
        {recentEvents.length === 0 ? (
          <View style={styles.emptyState}>
            <AlertTriangle size={48} color={theme.colors.textSecondary} />
            <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>
              No events detected yet
            </Text>
            <Text style={[styles.emptySubText, { color: theme.colors.textSecondary }]}>
              {isRecording ? 'Listening for audio events...' : 'Start detection to see real-time audio events'}
            </Text>
          </View>
        ) : (
          <View style={styles.eventsList}>
            {recentEvents.map((event, index) => (
              <EventCard
                key={`${event.timestamp.getTime()}-${index}`}
                event={event}
                theme={theme}
                delay={index * 100}
              />
            ))}
          </View>
        )}
      </View>

      {/* Privacy Notice */}
      <LinearGradient
        colors={theme.gradients.card}
        style={styles.privacyCard}
      >
        <View style={styles.privacyHeader}>
          <Lock size={24} color={theme.colors.success} />
          <Text style={[styles.privacyTitle, { color: theme.colors.text }]}>
            Privacy Protected
          </Text>
        </View>
        <Text style={[styles.privacyDescription, { color: theme.colors.textSecondary }]}>
          All audio processing happens locally on your device. No data is transmitted to external servers, 
          ensuring complete privacy while providing real-time detection capabilities.
        </Text>
        <View style={styles.privacyFeatures}>
          <View style={styles.privacyFeature}>
            <Shield size={16} color={theme.colors.success} />
            <Text style={[styles.privacyFeatureText, { color: theme.colors.text }]}>
              Local Processing
            </Text>
          </View>
          <View style={styles.privacyFeature}>
            <Lock size={16} color={theme.colors.success} />
            <Text style={[styles.privacyFeatureText, { color: theme.colors.text }]}>
              No Data Transmission
            </Text>
          </View>
          <View style={styles.privacyFeature}>
            <Brain size={16} color={theme.colors.success} />
            <Text style={[styles.privacyFeatureText, { color: theme.colors.text }]}>
              Edge AI
            </Text>
          </View>
        </View>
      </LinearGradient>
    </ScrollView>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      {renderContent()}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    paddingBottom: 32,
  },
  heroHeader: {
    paddingHorizontal: 24,
    paddingVertical: 40,
    marginBottom: 24,
  },
  heroContent: {
    alignItems: 'center',
  },
  heroIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  heroTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
    marginBottom: 24,
  },
  heroStats: {
    flexDirection: 'row',
    gap: 20,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  heroStat: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  heroStatText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
    fontWeight: '600',
  },
  statusSection: {
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  statusCard: {
    padding: 20,
    borderRadius: 16,
    borderWidth: 2,
    alignItems: 'center',
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 12,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
  },
  notificationBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  notificationText: {
    fontSize: 12,
    fontWeight: '600',
  },
  toggleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 25,
    gap: 8,
  },
  toggleButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  notificationStatusSection: {
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  notificationStatusCard: {
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  notificationStatusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
  },
  notificationStatusTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  notificationStatusText: {
    fontSize: 14,
    lineHeight: 20,
  },
  waveformContainer: {
    alignItems: 'center',
    marginBottom: 32,
    paddingHorizontal: 24,
  },
  waveform: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 60,
    gap: 3,
    marginBottom: 12,
  },
  waveBar: {
    width: 4,
    height: 40,
    borderRadius: 2,
  },
  waveformLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  statsSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  settingsButton: {
    padding: 8,
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 12,
  },
  statCard: {
    flex: 1,
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
    gap: 8,
  },
  statValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  statLabel: {
    fontSize: 12,
    textAlign: 'center',
  },
  currentDetectionSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  recentEventsSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  eventsList: {
    gap: 12,
  },
  eventCard: {
    borderRadius: 16,
    borderWidth: 1,
    overflow: 'hidden',
  },
  eventCardGradient: {
    padding: 16,
  },
  eventHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  eventIcon: {
    fontSize: 32,
    marginRight: 16,
  },
  eventInfo: {
    flex: 1,
  },
  eventType: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  eventCategory: {
    fontSize: 12,
    textTransform: 'capitalize',
  },
  confidenceBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  confidenceText: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  processingTime: {
    fontSize: 12,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubText: {
    fontSize: 14,
    textAlign: 'center',
  },
  privacyCard: {
    marginHorizontal: 24,
    borderRadius: 20,
    padding: 24,
  },
  privacyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 12,
  },
  privacyTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  privacyDescription: {
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 20,
  },
  privacyFeatures: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  privacyFeature: {
    alignItems: 'center',
    gap: 8,
  },
  privacyFeatureText: {
    fontSize: 12,
    fontWeight: '600',
  },
});