import React, { useEffect, useRef } from 'react';
import { 
  View, 
  Text, 
  ScrollView, 
  TouchableOpacity, 
  StyleSheet, 
  Dimensions,
  Animated,
  Platform
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useTheme } from '@/components/ThemeProvider';
import { Mic, ChartBar as BarChart3, Settings as SettingsIcon, BookOpen, Zap, Shield, Activity, Brain, Waves, Play, TrendingUp, Users, Globe, Sparkles } from 'lucide-react-native';
import { LinearGradient } from 'expo-linear-gradient';

const { width, height } = Dimensions.get('window');

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

const FloatingParticle = ({ delay = 0 }: { delay?: number }) => {
  const { theme } = useTheme();
  const animatedValue = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(0)).current;
  const opacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const startAnimation = () => {
      Animated.loop(
        Animated.sequence([
          Animated.timing(animatedValue, {
            toValue: 1,
            duration: 3000 + Math.random() * 2000,
            useNativeDriver: true,
          }),
          Animated.timing(animatedValue, {
            toValue: 0,
            duration: 3000 + Math.random() * 2000,
            useNativeDriver: true,
          }),
        ])
      ).start();

      Animated.loop(
        Animated.sequence([
          Animated.timing(translateY, {
            toValue: -50,
            duration: 4000,
            useNativeDriver: true,
          }),
          Animated.timing(translateY, {
            toValue: 50,
            duration: 4000,
            useNativeDriver: true,
          }),
        ])
      ).start();

      Animated.loop(
        Animated.sequence([
          Animated.timing(opacity, {
            toValue: 0.6,
            duration: 2000,
            useNativeDriver: true,
          }),
          Animated.timing(opacity, {
            toValue: 0.2,
            duration: 2000,
            useNativeDriver: true,
          }),
        ])
      ).start();
    };

    setTimeout(startAnimation, delay);
  }, []);

  const scale = animatedValue.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [0.5, 1.2, 0.8],
  });

  return (
    <Animated.View
      style={[
        styles.particle,
        {
          backgroundColor: theme.colors.primary,
          transform: [
            { scale },
            { translateY },
          ],
          opacity,
        },
      ]}
    />
  );
};

const FeatureCard = ({ 
  icon, 
  label, 
  desc, 
  onPress, 
  theme, 
  gradient = false,
  delay = 0 
}: any) => {
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
    onPress();
  };

  return (
    <AnimatedTouchableOpacity
      style={[
        styles.featureCard,
        {
          transform: [{ scale: scaleAnim }],
          opacity: opacityAnim,
        },
      ]}
      onPress={handlePress}
      activeOpacity={0.9}
    >
      {gradient ? (
        <LinearGradient
          colors={theme.gradients.card}
          style={styles.featureCardGradient}
        >
          <FeatureCardContent icon={icon} label={label} desc={desc} theme={theme} />
        </LinearGradient>
      ) : (
        <View style={[styles.featureCardContent, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}>
          <FeatureCardContent icon={icon} label={label} desc={desc} theme={theme} />
        </View>
      )}
    </AnimatedTouchableOpacity>
  );
};

const FeatureCardContent = ({ icon, label, desc, theme }: any) => (
  <>
    <View style={[styles.featureIconContainer, { backgroundColor: theme.colors.primary + '20' }]}>
      {icon}
    </View>
    <Text style={[styles.featureCardLabel, { color: theme.colors.text }]}>{label}</Text>
    <Text style={[styles.featureCardDesc, { color: theme.colors.textSecondary }]}>{desc}</Text>
  </>
);

const StatCard = ({ icon, value, label, color, delay = 0 }: any) => {
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
    outputRange: [30, 0],
  });

  return (
    <Animated.View
      style={[
        styles.statCard,
        {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          transform: [{ translateY }],
          opacity: animatedValue,
        },
      ]}
    >
      <View style={[styles.statIcon, { backgroundColor: color + '20' }]}>
        {React.cloneElement(icon, { size: 20, color })}
      </View>
      <Text style={[styles.statValue, { color: theme.colors.text }]}>{value}</Text>
      <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>{label}</Text>
    </Animated.View>
  );
};

import { useLiveEvents } from '@/hooks/useLiveEvents';

export default function DashboardScreen() {
  const { stats: liveStats } = useLiveEvents();
  const { theme, isDark } = useTheme();
  const router = useRouter();
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const waveAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Pulse animation for the main action button
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.05,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    ).start();

    // Wave animation
    Animated.loop(
      Animated.timing(waveAnim, {
        toValue: 1,
        duration: 3000,
        useNativeDriver: true,
      })
    ).start();
  }, []);

  const features = [
    {
      icon: <Mic size={28} color={theme.colors.primary} />,
      label: 'Live Detection',
      desc: 'Real-time audio classification',
      path: '/(tabs)/detection',
      gradient: true,
    },
    {
      icon: <BarChart3 size={28} color={theme.colors.success} />,
      label: 'Analytics',
      desc: 'Event history and insights',
      path: '/(tabs)/events',
    },
    {
      icon: <Brain size={28} color={theme.colors.accent} />,
      label: 'AI Learning',
      desc: 'Federated learning system',
      path: '/(tabs)/learning',
    },
    {
      icon: <SettingsIcon size={28} color={theme.colors.info} />,
      label: 'Settings',
      desc: 'Customize your experience',
      path: '/(tabs)/settings',
    },
  ];

  const stats = [
    { icon: <Activity />, value: `${(liveStats.detectionAccuracy ?? liveStats.avgConfidence ?? 0) * 100 > 0 ? ((liveStats.detectionAccuracy ?? liveStats.avgConfidence) * 100).toFixed(1) : '0.0'}%`, label: 'Accuracy', color: theme.colors.success },
    { icon: <Zap />, value: typeof liveStats.avgLatency === 'number' && liveStats.avgLatency > 0 ? `${liveStats.avgLatency.toFixed(1)}ms` : 'N/A', label: 'Latency', color: theme.colors.primary },
    { icon: <Shield />, value: '100%', label: 'Private', color: theme.colors.accent },
    { icon: <Globe />, value: `${liveStats.totalEvents ?? 0}`, label: 'Events', color: theme.colors.info },
  ];

  const renderContent = () => (
    <ScrollView 
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      showsVerticalScrollIndicator={false}
    >
      {/* Hero Section */}
      <LinearGradient
        colors={theme.gradients.hero}
        style={styles.heroSection}
      >
        {/* Floating Particles */}
        <View style={styles.particleContainer}>
          <FloatingParticle delay={0} />
          <FloatingParticle delay={1000} />
          <FloatingParticle delay={2000} />
        </View>

        <View style={styles.heroContent}>
          <Animated.View style={[styles.heroIcon, { transform: [{ scale: pulseAnim }] }]}>
            <Waves size={40} color="white" />
          </Animated.View>
          
          <Text style={styles.heroTitle}>AudioSense</Text>
          <Text style={styles.heroSubtitle}>
            Privacy-First AI Audio Detection
          </Text>
          
          <View style={styles.heroFeatures}>
            <View style={styles.heroFeature}>
              <Shield size={16} color="rgba(255, 255, 255, 0.8)" />
              <Text style={styles.heroFeatureText}>100% Local Processing</Text>
            </View>
            <View style={styles.heroFeature}>
              <Zap size={16} color="rgba(255, 255, 255, 0.8)" />
              <Text style={styles.heroFeatureText}>Real-time Detection</Text>
            </View>
          </View>

          <TouchableOpacity
            style={styles.mainAction}
            onPress={() => router.push('/(tabs)/detection')}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={['rgba(255, 255, 255, 0.2)', 'rgba(255, 255, 255, 0.1)']}
              style={styles.mainActionGradient}
            >
              <Play size={24} color="white" />
              <Text style={styles.mainActionText}>Start Detection</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </LinearGradient>

      {/* Stats Section */}
      <View style={styles.statsSection}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
          Performance Metrics
        </Text>
        <View style={styles.statsGrid}>
          {stats.map((stat, index) => (
            <StatCard
              key={stat.label}
              icon={stat.icon}
              value={stat.value}
              label={stat.label}
              color={stat.color}
              delay={index * 200}
            />
          ))}
        </View>
      </View>

      {/* Features Section */}
      <View style={styles.featuresSection}>
        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>
            Explore Features
          </Text>
          <Sparkles size={24} color={theme.colors.primary} />
        </View>
        
        <View style={styles.featureGrid}>
          {features.map((feature, index) => (
            <FeatureCard
              key={feature.label}
              icon={feature.icon}
              label={feature.label}
              desc={feature.desc}
              onPress={() => router.push(feature.path as any)}
              theme={theme}
              gradient={feature.gradient}
              delay={index * 150}
            />
          ))}
        </View>
      </View>

      {/* Technology Section */}
      <LinearGradient
        colors={theme.gradients.card}
        style={styles.techSection}
      >
        <View style={styles.techHeader}>
          <Brain size={28} color={theme.colors.primary} />
          <Text style={[styles.techTitle, { color: theme.colors.text }]}>
            Tech & Features
          </Text>
        </View>
        
        <Text style={[styles.techDescription, { color: theme.colors.textSecondary }]}>
          This dashboard is built with React Native (Expo), expo-av for audio, Animated API for smooth UI, and lucide-react-native icons. All audio processing is local (privacy-first). Includes event detection, analytics and customizable settings – all implemented with real code.
        </Text>

        <View style={[styles.techFeatures, { justifyContent: 'space-between' }]}> 
          <View style={styles.techFeature}>
            <View style={[styles.techFeatureIcon, { backgroundColor: theme.colors.success + '20' }]}> 
              <Shield size={16} color={theme.colors.success} />
            </View>
            <Text style={[styles.techFeatureText, { color: theme.colors.text }]}>Edge AI</Text>
          </View>
          <View style={styles.techFeature}>
            <View style={[styles.techFeatureIcon, { backgroundColor: theme.colors.primary + '20' }]}> 
              <Waves size={16} color={theme.colors.primary} />
            </View>
            <Text style={[styles.techFeatureText, { color: theme.colors.text }]}>Acoustic</Text>
          </View>
          <View style={styles.techFeature}>
            <View style={[styles.techFeatureIcon, { backgroundColor: theme.colors.accent + '20' }]}> 
              <Users size={16} color={theme.colors.accent} />
            </View>
            <Text style={[styles.techFeatureText, { color: theme.colors.text }]}>Smart Env.</Text>
          </View>
        </View>
      </LinearGradient>
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
  heroSection: {
    minHeight: height * 0.6,
    paddingHorizontal: 24,
    paddingVertical: 40,
    justifyContent: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  particleContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  particle: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    top: '20%',
    left: '10%',
  },
  heroContent: {
    alignItems: 'center',
    zIndex: 1,
  },
  heroIcon: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  heroTitle: {
    fontSize: 42,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 12,
    textAlign: 'center',
  },
  heroSubtitle: {
    fontSize: 18,
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 24,
  },
  heroFeatures: {
    flexDirection: 'row',
    gap: 24,
    marginBottom: 40,
  },
  heroFeature: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  heroFeatureText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    fontWeight: '600',
  },
  mainAction: {
    borderRadius: 25,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  mainActionGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    paddingHorizontal: 32,
    gap: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  mainActionText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 18,
  },
  statsSection: {
    paddingHorizontal: 24,
    paddingVertical: 32,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: (width - 60) / 2,
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
  },
  statIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    textAlign: 'center',
  },
  featuresSection: {
    paddingHorizontal: 24,
    paddingBottom: 32,
  },
  featureGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  featureCard: {
    flex: 1,
    minWidth: (width - 64) / 2,
  },
  featureCardGradient: {
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    minHeight: 160,
    justifyContent: 'center',
  },
  featureCardContent: {
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    borderWidth: 1,
    minHeight: 160,
    justifyContent: 'center',
  },
  featureIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  featureCardLabel: {
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 8,
    textAlign: 'center',
  },
  featureCardDesc: {
    fontSize: 13,
    textAlign: 'center',
    lineHeight: 18,
  },
  techSection: {
    marginHorizontal: 24,
    borderRadius: 24,
    padding: 28,
    marginBottom: 32,
  },
  techHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 12,
  },
  techTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  techDescription: {
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 24,
  },
  techFeatures: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  techFeature: {
    alignItems: 'center',
    gap: 8,
  },
  techFeatureIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  techFeatureText: {
    fontSize: 12,
    fontWeight: '600',
  },
});