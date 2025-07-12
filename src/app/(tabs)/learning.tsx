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
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Brain, Users, Shield, TrendingUp, Zap, Globe, Activity, Award, Target, CircleCheck as CheckCircle, Circle as XCircle, Upload, Download, Lock, Eye, Settings, Play, Pause, RotateCcw, Info } from 'lucide-react-native';
import { useTheme } from '@/components/ThemeProvider';

const { width, height } = Dimensions.get('window');

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

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

const TrainingSession = ({ session, theme, onStart, onStop, isActive }: any) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const progressAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (isActive) {
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
  }, [isActive]);

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

    if (isActive) {
      onStop();
    } else {
      onStart();
    }
  };

  return (
    <AnimatedTouchableOpacity
      style={[
        styles.trainingSession,
        {
          backgroundColor: theme.colors.card,
          borderColor: isActive ? theme.colors.primary : theme.colors.border,
          transform: [{ scale: scaleAnim }],
        },
      ]}
      onPress={handlePress}
      activeOpacity={0.8}
    >
      <LinearGradient
        colors={isActive ? [theme.colors.primary + '20', theme.colors.primary + '10'] : [theme.colors.card, theme.colors.surface]}
        style={styles.sessionGradient}
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
            {isActive ? (
              <Pause size={20} color={theme.colors.primary} />
            ) : (
              <Play size={20} color={theme.colors.primary} />
            )}
          </TouchableOpacity>
        </View>

        {isActive && (
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
      </LinearGradient>
    </AnimatedTouchableOpacity>
  );
};

export default function LearningScreen() {
  const { theme, isDark } = useTheme();
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [learningEnabled, setLearningEnabled] = useState(false);
  const [contributionLevel, setContributionLevel] = useState('Beginner');

  const metrics = [
    {
      icon: <Target />,
      value: '1,247',
      label: 'Contributions',
      color: theme.colors.primary,
      change: '+23%',
    },
    {
      icon: <Award />,
      value: '94.2%',
      label: 'Model Accuracy',
      color: theme.colors.success,
      change: '+2.1%',
    },
    {
      icon: <Users />,
      value: '12.5K',
      label: 'Global Participants',
      color: theme.colors.accent,
      change: '+156',
    },
    {
      icon: <Shield />,
      value: '100%',
      label: 'Privacy Score',
      color: theme.colors.info,
      change: 'Secure',
    },
  ];

  const trainingSessions = [
    {
      id: 'acoustic',
      title: 'Acoustic Event Detection',
      description: 'Improve detection of environmental sounds',
      icon: <Activity />,
      color: theme.colors.primary,
      accuracy: '94.2%',
      samples: '2.1K',
      rounds: '15',
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
    },
  ];

  const handleStartSession = (sessionId: string) => {
    if (!learningEnabled) {
      setShowPrivacyModal(true);
      return;
    }
    setActiveSession(sessionId);
  };

  const handleStopSession = () => {
    setActiveSession(null);
  };

  const enableLearning = () => {
    setLearningEnabled(true);
    setShowPrivacyModal(false);
  };

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
              onPress={enableLearning}
            >
              <Text style={styles.modalButtonText}>Enable Learning</Text>
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

  const renderContent = () => (
    <ScrollView 
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      showsVerticalScrollIndicator={false}
    >
      {/* Hero Header */}
      <LinearGradient
        colors={theme.gradients.hero}
        style={styles.heroHeader}
      >
        <View style={styles.heroContent}>
          <View style={styles.heroIcon}>
            <Brain size={32} color="white" />
          </View>
          <Text style={styles.heroTitle}>Federated Learning</Text>
          <Text style={styles.heroSubtitle}>
            Collaborate to improve AI while keeping your data private
          </Text>
          
          <View style={styles.heroStats}>
            <View style={styles.heroStat}>
              <Globe size={16} color="rgba(255, 255, 255, 0.8)" />
              <Text style={styles.heroStatText}>Global Network</Text>
            </View>
            <View style={styles.heroStat}>
              <Lock size={16} color="rgba(255, 255, 255, 0.8)" />
              <Text style={styles.heroStatText}>Privacy First</Text>
            </View>
          </View>
        </View>
      </LinearGradient>

      {/* Learning Status */}
      <View style={styles.statusSection}>
        <View style={[styles.statusCard, { 
          backgroundColor: learningEnabled ? theme.colors.success + '20' : theme.colors.error + '20',
          borderColor: learningEnabled ? theme.colors.success : theme.colors.error 
        }]}>
          <View style={styles.statusHeader}>
            {learningEnabled ? (
              <CheckCircle size={24} color={theme.colors.success} />
            ) : (
              <XCircle size={24} color={theme.colors.error} />
            )}
            <Text style={[styles.statusTitle, { color: theme.colors.text }]}>
              {learningEnabled ? 'Federated Learning Active' : 'Federated Learning Disabled'}
            </Text>
          </View>
          <Text style={[styles.statusDescription, { color: theme.colors.textSecondary }]}>
            {learningEnabled 
              ? 'You are contributing to global model improvements while maintaining privacy.'
              : 'Enable federated learning to help improve the AI model for everyone.'
            }
          </Text>
          {!learningEnabled && (
            <TouchableOpacity
              style={[styles.enableButton, { backgroundColor: theme.colors.primary }]}
              onPress={() => setShowPrivacyModal(true)}
            >
              <Text style={styles.enableButtonText}>Enable Learning</Text>
            </TouchableOpacity>
          )}
        </View>
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
      <LinearGradient
        colors={theme.gradients.card}
        style={styles.contributionCard}
      >
        <View style={styles.contributionHeader}>
          <Award size={24} color={theme.colors.accent} />
          <Text style={[styles.contributionTitle, { color: theme.colors.text }]}>
            Contribution Level: {contributionLevel}
          </Text>
        </View>
        
        <View style={styles.contributionProgress}>
          <View style={[styles.contributionBar, { backgroundColor: theme.colors.surface }]}>
            <View style={[styles.contributionFill, { 
              backgroundColor: theme.colors.accent,
              width: '65%' 
            }]} />
          </View>
          <Text style={[styles.contributionText, { color: theme.colors.textSecondary }]}>
            247 more contributions to reach Advanced level
          </Text>
        </View>

        <View style={styles.contributionBenefits}>
          <Text style={[styles.benefitsTitle, { color: theme.colors.text }]}>Benefits:</Text>
          <Text style={[styles.benefitsText, { color: theme.colors.textSecondary }]}>
            • Early access to new models{'\n'}
            • Detailed learning analytics{'\n'}
            • Community recognition
          </Text>
        </View>
      </LinearGradient>

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
            <TrainingSession
              key={session.id}
              session={session}
              theme={theme}
              onStart={() => handleStartSession(session.id)}
              onStop={handleStopSession}
              isActive={activeSession === session.id}
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

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: theme.colors.background }]}>
      {renderContent()}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  container: {
    flex: 1,
  },
  contentContainer: {
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
    gap: 24,
  },
  heroStat: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  heroStatText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    fontWeight: '600',
  },
  statusSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  statusCard: {
    padding: 20,
    borderRadius: 16,
    borderWidth: 2,
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
    marginBottom: 16,
  },
  enableButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  enableButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  metricsSection: {
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
    marginHorizontal: 24,
    borderRadius: 20,
    padding: 24,
    marginBottom: 32,
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
    marginBottom: 16,
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
  contributionBenefits: {
    gap: 8,
  },
  benefitsTitle: {
    fontSize: 14,
    fontWeight: '600',
  },
  benefitsText: {
    fontSize: 12,
    lineHeight: 18,
  },
  sessionsSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  sessionsList: {
    gap: 16,
  },
  trainingSession: {
    borderRadius: 16,
    borderWidth: 2,
    overflow: 'hidden',
  },
  sessionGradient: {
    padding: 20,
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
    paddingHorizontal: 24,
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