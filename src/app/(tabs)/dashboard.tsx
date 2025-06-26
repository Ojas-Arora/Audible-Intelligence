import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useTheme } from '@/components/ThemeProvider';
import { Mic, BarChart3, Settings as SettingsIcon, BookOpen } from 'lucide-react-native';
import { LinearGradient } from 'expo-linear-gradient';

const FeatureCard = ({ icon, label, desc, onPress, theme }: any) => (
  <TouchableOpacity style={[styles.featureCard, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]} onPress={onPress}>
    {icon}
    <Text style={[styles.featureCardLabel, { color: theme.colors.text }]}>{label}</Text>
    <Text style={[styles.featureCardDesc, { color: theme.colors.textSecondary }]}>{desc}</Text>
  </TouchableOpacity>
);

export default function DashboardScreen() {
  const { theme, isDark } = useTheme();
  const router = useRouter();

  const features = [
    {
      icon: <Mic size={28} color={theme.colors.primary} />,
      label: 'Detection',
      desc: 'Real-time audio classification',
      path: '/(tabs)/',
    },
    {
      icon: <BarChart3 size={28} color={theme.colors.success} />,
      label: 'Events',
      desc: 'Event history and analytics',
      path: '/(tabs)/events',
    },
    {
      icon: <SettingsIcon size={28} color={theme.colors.accent} />,
      label: 'Settings',
      desc: 'Customize your experience',
      path: '/(tabs)/settings',
    },
    {
      icon: <BookOpen size={28} color={theme.colors.info} />,
      label: 'Learning',
      desc: 'Federated Learning',
      path: '/(tabs)/learning',
    },
  ];

  const renderContent = () => (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.header}>
        <Text style={[styles.title, { color: theme.colors.text }]}>DCASE Acoustic Source & Event Detection</Text>
        <Text style={[styles.subtitle, { color: theme.colors.textSecondary }]}>
          Inspired by the DCASE Challenge, this project is a lightweight, privacy-preserving mobile/web app for real-time acoustic event detection and classification.
        </Text>
      </View>

      <View style={styles.featureGrid}>
        {features.map((feature) => (
          <FeatureCard
            key={feature.label}
            icon={feature.icon}
            label={feature.label}
            desc={feature.desc}
            onPress={() => router.push(feature.path as any)}
            theme={theme}
          />
        ))}
      </View>
    </ScrollView>
  );

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: isDark ? '#0f1123' : theme.colors.background }]}>
      {isDark ? (
        <LinearGradient colors={theme.gradients.background} style={styles.safeArea}>
          {renderContent()}
        </LinearGradient>
      ) : (
        renderContent()
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  container: {
    padding: 24,
  },
  header: {
    marginBottom: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 16,
    marginTop: 8,
    lineHeight: 24,
  },
  mainAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 18,
    borderRadius: 16,
    gap: 12,
    marginBottom: 24,
  },
  mainActionText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 18,
  },
  featureGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -8,
  },
  featureCard: {
    flexGrow: 1,
    flexBasis: '40%', // Ensure two cards per row
    margin: 8,
    padding: 20,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
  },
  featureCardLabel: {
    fontWeight: 'bold',
    fontSize: 16,
    marginTop: 16,
    textAlign: 'center',
  },
  featureCardDesc: {
    fontSize: 13,
    marginTop: 6,
    textAlign: 'center',
    opacity: 0.8,
  },
});