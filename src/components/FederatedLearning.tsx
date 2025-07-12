import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Animated,
  Modal,
  Platform,
  Alert,
} from 'react-native';
import { Brain, Users, Shield, TrendingUp, Zap, Globe, Activity, Award, Target, CircleCheck as CheckCircle, Circle as XCircle, Upload, Download, Lock, Eye, Settings, Play, Pause, RotateCcw, Info, Wifi, WifiOff } from 'lucide-react-native';
import { useTheme } from './ThemeProvider';
import FederatedLearningService from '@/services/FederatedLearningService';

const { width, height } = Dimensions.get('window');

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

interface LearningMetrics {
  totalContributions: number;
  accuracyImprovement: number;
  modelUpdatesReceived: number;
  contributionScore: number;
  learningStreak: number;
  privacyScore: number;
  globalParticipants: number;
  globalAccuracy: number;
}

interface TrainingSession {
  id: string;
  title: string;
  description: string;
  icon: React.ReactElement;
  color: string;
  accuracy: string;
  samples: string;
  rounds: string;
  isActive: boolean;
}

const MetricCard = ({ icon, value, label, color, change, delay = 0 }: any) => {
  const { theme } = useTheme();
  const animatedValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    setTimeout(() => {
      Animated.timing(animatedValue, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }).start();
    }, delay);
  }, []);

  const translateY = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [20, 0],
  });

  return (
    <Animated.View
      style={[
        styles.metricCard,
        {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          transform: [{ translateY }],
          opacity: animatedValue,
        },
      ]}
    >
      <View style={[styles.metricIcon, { backgroundColor: color + '20' }]}>
        {React.cloneElement(icon, { size: 20, color })}
      </View>
      <Text style={[styles.metricValue, { color: theme.colors.text }]}>{value}</Text>
      <Text style={[styles.metricLabel, { color: theme.colors.textSecondary }]}>{label}</Text>
      {change && (
        <View style={styles.metricChange}>
          <TrendingUp size={12} color={theme.colors.success} />
          <Text style={[styles.metricChangeText, { color: theme.colors.success }]}>{change}</Text>
        </View>
      )}
    </Animated.View>
  );
};

const TrainingSessionCard = ({ session, theme, onStart, onStop }: any) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (session.isActive) {
      Animated.loop(
        Animated.timing(progressAnim, {
          toValue: 1,
          duration: 3000,
          useNativeDriver: false,
        })
      ).start();
    } else {
      progressAnim.setValue(0);
    }
  }, [session.isActive]);

  const handlePress = () => {
    Animated.sequence([
      Animated.timing(scaleAnim, {
        toValue: 0.95,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      }),
    ]).start();

    if (session.isActive) {
      onStop(session.id);
    } else {
      onStart(session.id);
    }
  };

  return (
    <AnimatedTouchableOpacity
      style={[
        styles.trainingSession,
        {
          backgroundColor: theme.colors.card,
          borderColor: session.isActive ? theme.colors.primary : theme.colors.border,
          transform: [{ scale: scaleAnim }],
        },
      ]}
      onPress={handlePress}
      activeOpacity={0.8}
    >
      <View style={styles.sessionHeader}>
        <View style={[styles.sessionIcon, { backgroundColor: session.color + '20' }]}>
          {React.cloneElement(session.icon, { size: 24, color: session.color })}
        </View>
        <View style={styles.sessionInfo}>
          <Text style={[styles.sessionTitle, { color: theme.colors.text }]}>{session.title}</Text>
          <Text style={[styles.sessionDescription, { color: theme.colors.textSecondary }]}>
            {session.description}
          </Text>
        </View>
        <TouchableOpacity style={styles.sessionAction}>
          {session.isActive ? (
            <Pause size={20} color={theme.colors.primary} />
          ) : (
            <Play size={20} color={theme.colors.primary} />
          )}
        </TouchableOpacity>
      </View>

      {session.isActive && (
        <View style={styles.sessionProgress}>
          <View style={[styles.progressBar, { backgroundColor: theme.colors.surface }]}>
            <Animated.View
              style={[
                styles.progressFill,
                {
                  backgroundColor: theme.colors.primary,
                  width: progressAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: ['0%', '100%'],
                  }),
                },
              ]}
            />
          </View>
          <Text style={[styles.progressText, { color: theme.colors.textSecondary }]}>
            Training in progress...
          </Text>
        </View>
      )}

      <View style={styles.sessionStats}>
        <View style={styles.sessionStat}>
          <Text style={[styles.sessionStatValue, { color: theme.colors.text }]}>{session.accuracy}</Text>
          <Text style={[styles.sessionStatLabel, { color: theme.colors.textSecondary }]}>Accuracy</Text>
        </View>
        <View style={styles.sessionStat}>
          <Text style={[styles.sessionStatValue, { color: theme.colors.text }]}>{session.samples}</Text>
          <Text style={[styles.sessionStatLabel, { color: theme.colors.textSecondary }]}>Samples</Text>
        </View>
        <View style={styles.sessionStat}>
          <Text style={[styles.sessionStatValue, { color: theme.colors.text }]}>{session.rounds}</Text>
          <Text style={[styles.sessionStatLabel, { color: theme.colors.textSecondary }]}>Rounds</Text>
        </View>
      </View>
    </AnimatedTouchableOpacity>
  );
};

export const FederatedLearning: React.FC = () => {
  const { theme } = useTheme();
  const [isConnected, setIsConnected] = useState(false);
  const [isParticipant, setIsParticipant] = useState(false);
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [learningMetrics, setLearningMetrics] = useState<LearningMetrics>({
    totalContributions: 0,
    accuracyImprovement: 0,
    modelUpdatesReceived: 0,
    contributionScore: 0,
    learningStreak: 0,
    privacyScore: 100,
    globalParticipants: 0,
    globalAccuracy: 0,
  });
  const [activeSessions, setActiveSessions] = useState<Set<string>>(new Set());
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');

  const trainingSessions: TrainingSession[] = [
    {
      id: 'acoustic',
      title: 'Acoustic Event Detection',
      description: 'Improve detection of environmental sounds',
      icon: <Activity />,
      color: theme.colors.primary,
      accuracy: '94.2%',
      samples: '2.1K',
      rounds: '15',
      isActive: activeSessions.has('acoustic'),
    },
    {
      id: 'speech',
      title: 'Speech Recognition',
      description: 'Enhance speech pattern recognition',
      icon: <Brain />,
      color: theme.colors.success,
      accuracy: '91.8%',
      samples: '1.8K',
      rounds: '12',
      isActive: activeSessions.has('speech'),
    },
    {
      id: 'noise',
      title: 'Noise Classification',
      description: 'Better noise filtering and classification',
      icon: <Zap />,
      color: theme.colors.accent,
      accuracy: '89.5%',
      samples: '1.5K',
      rounds: '8',
      isActive: activeSessions.has('noise'),
    },
  ];

  useEffect(() => {
    initializeFederatedLearning();
    setupEventListeners();

    return () => {
      FederatedLearningService.disconnect();
    };
  }, []);

  const initializeFederatedLearning = async () => {
    try {
      setConnectionStatus('connecting');
      
      // Check if already a participant
      setIsParticipant(FederatedLearningService.isParticipant());
      
      // Connect to WebSocket
      const connected = await FederatedLearningService.connectWebSocket();
      setIsConnected(connected);
      setConnectionStatus(connected ? 'connected' : 'disconnected');
      
      // Load initial data
      if (connected) {
        await loadFederatedLearningData();
      }
    } catch (error) {
      console.error('Error initializing federated learning:', error);
      setConnectionStatus('disconnected');
    }
  };

  const setupEventListeners = () => {
    FederatedLearningService.on('connected', () => {
      setIsConnected(true);
      setConnectionStatus('connected');
    });

    FederatedLearningService.on('disconnected', () => {
      setIsConnected(false);
      setConnectionStatus('disconnected');
    });

    FederatedLearningService.on('joinedFederatedLearning', (data: any) => {
      setIsParticipant(true);
      updateMetricsFromGlobalModel(data.globalModel);
    });

    FederatedLearningService.on('globalModelUpdate', (data: any) => {
      updateMetricsFromGlobalModel(data.globalModel);
      showAlert('Model Updated!', `New global model version ${data.globalModel.version} is available with improved accuracy.`);
    });

    FederatedLearningService.on('participantUpdate', (data: any) => {
      setLearningMetrics(prev => ({
        ...prev,
        globalParticipants: data.participantCount,
      }));
    });

    FederatedLearningService.on('modelUpdateReceived', (data: any) => {
      setLearningMetrics(prev => ({
        ...prev,
        totalContributions: data.contributionCount,
        modelUpdatesReceived: prev.modelUpdatesReceived + 1,
        contributionScore: prev.contributionScore + 10,
      }));
    });
  };

  const loadFederatedLearningData = async () => {
    try {
      const status = await FederatedLearningService.getFederatedLearningStatus();
      updateMetricsFromGlobalModel(status.globalModel);
      setLearningMetrics(prev => ({
        ...prev,
        globalParticipants: status.participantCount,
      }));
    } catch (error) {
      console.error('Error loading federated learning data:', error);
    }
  };

  const updateMetricsFromGlobalModel = (globalModel: any) => {
    setLearningMetrics(prev => ({
      ...prev,
      globalAccuracy: globalModel.accuracy * 100,
      accuracyImprovement: prev.accuracyImprovement + 0.1,
    }));
  };

  const joinFederatedLearning = async () => {
    try {
      const deviceInfo = {
        platform: Platform.OS,
        version: Platform.Version,
        timestamp: new Date().toISOString(),
      };

      const result = await FederatedLearningService.joinFederatedLearning(deviceInfo);
      setIsParticipant(true);
      updateMetricsFromGlobalModel(result.globalModel);
      setShowPrivacyModal(false);
      
      showAlert('Welcome!', 'You have successfully joined the federated learning network. Your contributions will help improve the global model while keeping your data private.');
    } catch (error) {
      console.error('Error joining federated learning:', error);
      showAlert('Error', 'Failed to join federated learning. Please try again.');
    }
  };

  const startTrainingSession = async (sessionId: string) => {
    if (!isParticipant) {
      setShowPrivacyModal(true);
      return;
    }

    try {
      setActiveSessions(prev => new Set([...prev, sessionId]));
      
      // Simulate local training
      const trainingData = Array.from({ length: 100 }, (_, i) => ({ id: i, data: Math.random() }));
      const modelUpdate = await FederatedLearningService.simulateLocalTraining(trainingData);
      
      // Submit model update
      await FederatedLearningService.submitModelUpdate({
        modelWeights: modelUpdate.modelWeights,
        trainingMetrics: modelUpdate.trainingMetrics,
      });
      
      setActiveSessions(prev => {
        const newSet = new Set(prev);
        newSet.delete(sessionId);
        return newSet;
      });
      
      showAlert('Training Complete!', 'Your local model update has been submitted to the global federated learning network.');
    } catch (error) {
      console.error('Error in training session:', error);
      setActiveSessions(prev => {
        const newSet = new Set(prev);
        newSet.delete(sessionId);
        return newSet;
      });
      showAlert('Training Failed', 'There was an error during the training session. Please try again.');
    }
  };

  const stopTrainingSession = (sessionId: string) => {
    setActiveSessions(prev => {
      const newSet = new Set(prev);
      newSet.delete(sessionId);
      return newSet;
    });
  };

  const showAlert = (title: string, message: string) => {
    if (Platform.OS === 'web') {
      alert(`${title}\n\n${message}`);
    } else {
      Alert.alert(title, message);
    }
  };

  const getContributionLevel = (score: number) => {
    if (score >= 500) return { level: 'Expert', color: theme.colors.success, icon: '🏆' };
    if (score >= 200) return { level: 'Advanced', color: theme.colors.primary, icon: '🥇' };
    if (score >= 50) return { level: 'Intermediate', color: theme.colors.accent, icon: '🥈' };
    return { level: 'Beginner', color: theme.colors.textSecondary, icon: '🥉' };
  };

  const contributionLevel = getContributionLevel(learningMetrics.contributionScore);

  const metrics = [
    {
      icon: <Target />,
      value: learningMetrics.totalContributions.toString(),
      label: 'Contributions',
      color: theme.colors.primary,
      change: '+23%',
    },
    {
      icon: <Award />,
      value: `${learningMetrics.globalAccuracy.toFixed(1)}%`,
      label: 'Global Accuracy',
      color: theme.colors.success,
      change: '+2.1%',
    },
    {
      icon: <Users />,
      value: learningMetrics.globalParticipants.toLocaleString(),
      label: 'Participants',
      color: theme.colors.accent,
      change: '+156',
    },
    {
      icon: <Shield />,
      value: `${learningMetrics.privacyScore}%`,
      label: 'Privacy Score',
      color: theme.colors.info,
      change: 'Secure',
    },
  ];

  const PrivacyModal = () => (
    <Modal
      visible={showPrivacyModal}
      transparent
      animationType="fade"
      onRequestClose={() => setShowPrivacyModal(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={[styles.modalContent, { backgroundColor: theme.colors.card }]}>
          <View style={styles.modalHeader}>
            <Shield size={32} color={theme.colors.primary} />
            <Text style={[styles.modalTitle, { color: theme.colors.text }]}>
              Privacy-Preserving Learning
            </Text>
          </View>
          
          <Text style={[styles.modalText, { color: theme.colors.textSecondary }]}>
            Federated learning allows you to contribute to model improvements while keeping your data completely private:
          </Text>

          <View style={styles.privacyFeatures}>
            <View style={styles.privacyFeature}>
              <Lock size={16} color={theme.colors.success} />
              <Text style={[styles.privacyFeatureText, { color: theme.colors.text }]}>
                Your audio never leaves your device
              </Text>
            </View>
            <View style={styles.privacyFeature}>
              <Eye size={16} color={theme.colors.success} />
              <Text style={[styles.privacyFeatureText, { color: theme.colors.text }]}>
                Only model updates are shared
              </Text>
            </View>
            <View style={styles.privacyFeature}>
              <Shield size={16} color={theme.colors.success} />
              <Text style={[styles.privacyFeatureText, { color: theme.colors.text }]}>
                Differential privacy protection
              </Text>
            </View>
          </View>

          <View style={styles.modalActions}>
            <TouchableOpacity
              style={[styles.modalButton, { backgroundColor: theme.colors.primary }]}
              onPress={joinFederatedLearning}
            >
              <Text style={styles.modalButtonText}>Join Network</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modalButtonSecondary, { borderColor: theme.colors.border }]}
              onPress={() => setShowPrivacyModal(false)}
            >
              <Text style={[styles.modalButtonSecondaryText, { color: theme.colors.textSecondary }]}>
                Maybe Later
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  return (
    <ScrollView 
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      showsVerticalScrollIndicator={false}
    >
      {/* Connection Status */}
      <View style={[
        styles.connectionStatus,
        { 
          backgroundColor: isConnected ? theme.colors.success + '20' : theme.colors.error + '20',
          borderColor: isConnected ? theme.colors.success : theme.colors.error 
        }
      ]}>
        <View style={styles.connectionHeader}>
          {isConnected ? (
            <Wifi size={20} color={theme.colors.success} />
          ) : (
            <WifiOff size={20} color={theme.colors.error} />
          )}
          <Text style={[styles.connectionText, { color: theme.colors.text }]}>
            {connectionStatus === 'connecting' ? 'Connecting...' : 
             isConnected ? 'Connected to FL Network' : 'Disconnected'}
          </Text>
        </View>
        {!isParticipant && isConnected && (
          <TouchableOpacity
            style={[styles.joinButton, { backgroundColor: theme.colors.primary }]}
            onPress={() => setShowPrivacyModal(true)}
          >
            <Text style={styles.joinButtonText}>Join Network</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Learning Status */}
      <View style={[styles.statusCard, { 
        backgroundColor: isParticipant ? theme.colors.success + '20' : theme.colors.warning + '20',
        borderColor: isParticipant ? theme.colors.success : theme.colors.warning 
      }]}>
        <View style={styles.statusHeader}>
          {isParticipant ? (
            <CheckCircle size={24} color={theme.colors.success} />
          ) : (
            <XCircle size={24} color={theme.colors.warning} />
          )}
          <Text style={[styles.statusTitle, { color: theme.colors.text }]}>
            {isParticipant ? 'Federated Learning Active' : 'Join Federated Learning'}
          </Text>
        </View>
        <Text style={[styles.statusDescription, { color: theme.colors.textSecondary }]}>
          {isParticipant 
            ? 'You are contributing to global model improvements while maintaining privacy.'
            : 'Join the global network to help improve AI models while keeping your data private.'
          }
        </Text>
      </View>

      {/* Metrics */}
      <View style={styles.metricsSection}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
          Learning Metrics
        </Text>
        <View style={styles.metricsGrid}>
          {metrics.map((metric, index) => (
            <MetricCard
              key={metric.label}
              icon={metric.icon}
              value={metric.value}
              label={metric.label}
              color={metric.color}
              change={metric.change}
              delay={index * 100}
            />
          ))}
        </View>
      </View>

      {/* Contribution Level */}
      <View style={[styles.contributionCard, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}>
        <View style={styles.contributionHeader}>
          <Award size={24} color={contributionLevel.color} />
          <Text style={[styles.contributionTitle, { color: theme.colors.text }]}>
            {contributionLevel.level} Contributor {contributionLevel.icon}
          </Text>
        </View>
        
        <View style={styles.contributionProgress}>
          <View style={[styles.contributionBar, { backgroundColor: theme.colors.surface }]}>
            <View style={[styles.contributionFill, { 
              backgroundColor: contributionLevel.color,
              width: `${Math.min((learningMetrics.contributionScore / 500) * 100, 100)}%` 
            }]} />
          </View>
          <Text style={[styles.contributionText, { color: theme.colors.textSecondary }]}>
            {500 - learningMetrics.contributionScore > 0 
              ? `${500 - learningMetrics.contributionScore} more points to reach Expert level`
              : 'Expert level achieved!'
            }
          </Text>
        </View>
      </View>

      {/* Training Sessions */}
      <View style={styles.sessionsSection}>
        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
            Training Sessions
          </Text>
          <TouchableOpacity style={styles.settingsButton}>
            <Settings size={20} color={theme.colors.textSecondary} />
          </TouchableOpacity>
        </View>

        <View style={styles.sessionsList}>
          {trainingSessions.map((session) => (
            <TrainingSessionCard
              key={session.id}
              session={session}
              theme={theme}
              onStart={startTrainingSession}
              onStop={stopTrainingSession}
            />
          ))}
        </View>
      </View>

      {/* How It Works */}
      <View style={styles.howItWorksSection}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
          How Federated Learning Works
        </Text>
        
        <View style={styles.stepsContainer}>
          <View style={styles.step}>
            <View style={[styles.stepNumber, { backgroundColor: theme.colors.primary }]}>
              <Text style={styles.stepNumberText}>1</Text>
            </View>
            <View style={styles.stepContent}>
              <Text style={[styles.stepTitle, { color: theme.colors.text }]}>Local Training</Text>
              <Text style={[styles.stepDescription, { color: theme.colors.textSecondary }]}>
                Your device trains on local data without sharing it
              </Text>
            </View>
          </View>

          <View style={styles.step}>
            <View style={[styles.stepNumber, { backgroundColor: theme.colors.success }]}>
              <Text style={styles.stepNumberText}>2</Text>
            </View>
            <View style={styles.stepContent}>
              <Text style={[styles.stepTitle, { color: theme.colors.text }]}>Model Updates</Text>
              <Text style={[styles.stepDescription, { color: theme.colors.textSecondary }]}>
                Only encrypted model improvements are shared
              </Text>
            </View>
          </View>

          <View style={styles.step}>
            <View style={[styles.stepNumber, { backgroundColor: theme.colors.accent }]}>
              <Text style={styles.stepNumberText}>3</Text>
            </View>
            <View style={styles.stepContent}>
              <Text style={[styles.stepTitle, { color: theme.colors.text }]}>Global Improvement</Text>
              <Text style={[styles.stepDescription, { color: theme.colors.textSecondary }]}>
                Everyone benefits from the improved global model
              </Text>
            </View>
          </View>
        </View>
      </View>

      <PrivacyModal />
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  contentContainer: {
    paddingBottom: 32,
  },
  connectionStatus: {
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    marginBottom: 20,
  },
  connectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  connectionText: {
    fontSize: 16,
    fontWeight: '600',
  },
  joinButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  joinButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },
  statusCard: {
    padding: 20,
    borderRadius: 16,
    borderWidth: 2,
    marginBottom: 24,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    gap: 12,
  },
  statusTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  statusDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
  metricsSection: {
    marginBottom: 24,
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
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricCard: {
    flex: 1,
    minWidth: (width - 60) / 2,
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
  },
  metricIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 4,
  },
  metricChange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metricChangeText: {
    fontSize: 10,
    fontWeight: '600',
  },
  contributionCard: {
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    marginBottom: 24,
  },
  contributionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 12,
  },
  contributionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  contributionProgress: {
    marginBottom: 8,
  },
  contributionBar: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  contributionFill: {
    height: '100%',
    borderRadius: 4,
  },
  contributionText: {
    fontSize: 12,
  },
  sessionsSection: {
    marginBottom: 24,
  },
  sessionsList: {
    gap: 16,
  },
  trainingSession: {
    padding: 20,
    borderRadius: 16,
    borderWidth: 2,
  },
  sessionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  sessionIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  sessionInfo: {
    flex: 1,
  },
  sessionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  sessionDescription: {
    fontSize: 14,
  },
  sessionAction: {
    padding: 8,
  },
  sessionProgress: {
    marginBottom: 16,
  },
  progressBar: {
    height: 4,
    borderRadius: 2,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 12,
  },
  sessionStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  sessionStat: {
    alignItems: 'center',
  },
  sessionStatValue: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  sessionStatLabel: {
    fontSize: 12,
  },
  howItWorksSection: {
    marginBottom: 24,
  },
  stepsContainer: {
    gap: 20,
  },
  step: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 16,
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepNumberText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  stepDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    borderRadius: 20,
    padding: 24,
    width: '100%',
    maxWidth: 400,
    maxHeight: height * 0.8,
  },
  modalHeader: {
    alignItems: 'center',
    marginBottom: 20,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 12,
    textAlign: 'center',
  },
  modalText: {
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 20,
    textAlign: 'center',
  },
  privacyFeatures: {
    gap: 12,
    marginBottom: 24,
  },
  privacyFeature: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  privacyFeatureText: {
    fontSize: 14,
    flex: 1,
  },
  modalActions: {
    gap: 12,
  },
  modalButton: {
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  modalButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  modalButtonSecondary: {
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
  },
  modalButtonSecondaryText: {
    fontSize: 16,
    fontWeight: '600',
  },
});