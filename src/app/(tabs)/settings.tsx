import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Settings as SettingsIcon, Volume2, Shield, Bell, Mic, Database, Info, ChevronRight, TriangleAlert as AlertTriangle, Zap, Moon, Smartphone, RotateCcw } from 'lucide-react-native';
import { useTheme } from '@/components/ThemeProvider';
import { useAppSettings } from '@/hooks/useAppSettings';
import CommunitySlider from '@react-native-community/slider';

// Web-compatible slider to avoid findDOMNode error with React 19
const WebSlider = (props: any) => {
  return (
    <input
      type="range"
      min={props.minimumValue}
      max={props.maximumValue}
      step={0.01} // A smaller step for smoother sliding
      value={props.value}
      onChange={(e) => props.onValueChange(parseFloat(e.target.value))}
      style={{
        width: '100%',
        height: '20px',
        accentColor: props.thumbTintColor || props.minimumTrackTintColor,
      }}
    />
  );
};

const Slider = Platform.OS === 'web' ? WebSlider : CommunitySlider;

export default function SettingsScreen() {
  const { theme, isDark, toggleTheme } = useTheme();
  const { settings, updateSetting, resetSettings, isLoading } = useAppSettings();
  const [hoveredBox, setHoveredBox] = useState<string | null>(null);

  const Divider = () => <View style={{ height: 1, backgroundColor: theme.colors.border }} />;

  const handleResetSettings = () => {
    Alert.alert(
      'Reset Settings',
      'Are you sure you want to reset all settings to default values?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Reset', 
          style: 'destructive',
          onPress: resetSettings
        }
      ]
    );
  };

  const showAlert = (title: string, message: string) => {
    if (Platform.OS === 'web') {
      // Web fallback
      alert(`${title}\n\n${message}`);
    } else {
      Alert.alert(title, message);
    }
  };

  const SettingItem = ({ 
    icon, 
    title, 
    description, 
    value, 
    onValueChange, 
    iconColor = theme.colors.textSecondary,
    showArrow = false,
    onPress,
    children
  }: {
    icon: React.ReactNode;
    title: string;
    description?: string;
    value?: boolean;
    onValueChange?: (value: boolean) => void;
    iconColor?: string;
    showArrow?: boolean;
    onPress?: () => void;
    children?: React.ReactNode;
  }) => (
    <TouchableOpacity 
      style={[styles.settingItem, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}
      activeOpacity={showArrow || onPress ? 0.6 : 1}
      onPress={onPress}
    >
      <View style={styles.settingContent}>
        <View style={[styles.settingIcon, { backgroundColor: iconColor + '20' }]}>
          {React.cloneElement(icon as React.ReactElement, { 
            size: 20, 
            color: iconColor 
          })}
        </View>
        
        <View style={styles.settingText}>
          <Text style={[styles.settingTitle, { color: theme.colors.text }]}>{title}</Text>
          {description && (
            <Text style={[styles.settingDescription, { color: theme.colors.textSecondary }]}>{description}</Text>
          )}
          {children}
        </View>

        {onValueChange ? (
          <Switch
            value={value}
            onValueChange={onValueChange}
            thumbColor={value ? theme.colors.primary : theme.colors.surface}
            trackColor={{ false: theme.colors.surface, true: theme.colors.primary + '33' }}
          />
        ) : showArrow && (
          <ChevronRight size={20} color={theme.colors.textSecondary} />
        )}
      </View>
    </TouchableOpacity>
  );

  const renderContent = () => (
    <ScrollView contentContainerStyle={styles.content}>
      {/* Hero Section */}
      <View style={styles.header}>
        <SettingsIcon size={40} color={theme.colors.primary} />
        <Text style={[styles.title, { color: theme.colors.text }]}>Settings</Text>
        <Text style={[styles.subtitle, { color: theme.colors.textSecondary }]}>
          Customize your experience. All settings are applied instantly.
        </Text>
      </View>

      {/* General Settings Card */}
      <View style={[styles.card, { backgroundColor: theme.colors.card, borderColor: theme.colors.border }]}>
        <SettingItem
          icon={<Bell />}
          title="Enable Notifications"
          value={settings.notifications}
          onValueChange={v => updateSetting('notifications', v)}
          iconColor={theme.colors.primary}
        />
        <Divider />
        <SettingItem
          icon={<Volume2 />}
          title="Vibrate on Event"
          value={settings.hapticFeedback}
          onValueChange={v => updateSetting('hapticFeedback', v)}
          iconColor={theme.colors.primary}
        />
        <Divider />
        <SettingItem
          icon={<Moon />}
          title="Dark Mode"
          value={isDark}
          onValueChange={toggleTheme}
          iconColor={theme.colors.primary}
        />
      </View>

      {/* Detection Settings */}
      <View
        style={[styles.sectionCard, {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.colors.primary,
        }]}
      >
        <Text style={[styles.sectionHeader, { color: theme.colors.primary }]}>Detection</Text>
        <SettingItem
          icon={<Mic />}
          title="Auto Detection"
          description="Automatically start detection when app opens"
          value={settings.autoDetection}
          onValueChange={(value) => updateSetting('autoDetection', value)}
          iconColor={theme.colors.success}
        />
        <SettingItem
          icon={<Zap />}
          title="High Sensitivity"
          description="Detect quieter sounds with lower confidence threshold"
          value={settings.highSensitivity}
          onValueChange={(value) => updateSetting('highSensitivity', value)}
          iconColor={theme.colors.accent}
        />
        <SettingItem
          icon={<Smartphone />}
          title="Background Processing"
          description="Continue detection when app is in background"
          value={settings.backgroundProcessing}
          onValueChange={(value) => updateSetting('backgroundProcessing', value)}
          iconColor={theme.colors.secondary}
        />
        <SettingItem
          icon={<Volume2 />}
          title="Sensitivity Level"
          description={`Current: ${Math.round(settings.sensitivity * 100)}%`}
          iconColor={theme.colors.primary}
        >
          <View style={styles.sliderContainer}>
            <Slider
              style={styles.slider}
              minimumValue={0.1}
              maximumValue={1.0}
              value={settings.sensitivity}
              onValueChange={(value) => updateSetting('sensitivity', value)}
              minimumTrackTintColor={theme.colors.primary}
              maximumTrackTintColor={theme.colors.surface}
              thumbTintColor={theme.colors.primary}
            />
          </View>
        </SettingItem>
      </View>

      {/* Notifications & Feedback */}
      <View
        style={[styles.sectionCard, {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.colors.primary,
        }]}
      >
        <Text style={[styles.sectionHeader, { color: theme.colors.primary }]}>Notifications & Feedback</Text>
        <SettingItem
          icon={<Bell />}
          title="Event Notifications"
          description="Get notified when events are detected"
          value={settings.notifications}
          onValueChange={(value) => updateSetting('notifications', value)}
          iconColor={theme.colors.info}
        />
        <SettingItem
          icon={<Volume2 />}
          title="Haptic Feedback"
          description="Vibrate on event detection (mobile only)"
          value={settings.hapticFeedback}
          onValueChange={(value) => updateSetting('hapticFeedback', value)}
          iconColor={theme.colors.error}
        />
      </View>

      {/* Privacy & Data */}
      <View
        style={[styles.sectionCard, {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.colors.primary,
        }]}
      >
        <Text style={[styles.sectionHeader, { color: theme.colors.primary }]}>Privacy & Data</Text>
        <SettingItem
          icon={<Shield />}
          title="On-Device Processing"
          description="All audio processing happens locally"
          iconColor={theme.colors.success}
          showArrow
          onPress={() => showAlert('Privacy Info', 'All audio processing happens locally on your device. No data is transmitted to external servers.')}
        />
        <SettingItem
          icon={<Database />}
          title="Save Detections"
          description="Store detection history locally"
          value={settings.saveDetections}
          onValueChange={(value) => updateSetting('saveDetections', value)}
          iconColor={theme.colors.secondary}
        />
        <SettingItem
          icon={<AlertTriangle />}
          title="Analytics Collection"
          description="Help improve detection accuracy (anonymous)"
          value={settings.dataCollection}
          onValueChange={(value) => updateSetting('dataCollection', value)}
          iconColor={theme.colors.warning}
        />
      </View>

      {/* Advanced Settings */}
      <View
        style={[styles.sectionCard, {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.colors.primary,
        }]}
      >
        <Text style={[styles.sectionHeader, { color: theme.colors.primary }]}>Advanced</Text>
        <SettingItem
          icon={<Zap />}
          title="Confidence Threshold"
          description={`Minimum confidence: ${Math.round(settings.confidenceThreshold * 100)}%`}
          iconColor={theme.colors.accent}
        >
          <View style={styles.sliderContainer}>
            <Slider
              style={styles.slider}
              minimumValue={0.3}
              maximumValue={0.9}
              value={settings.confidenceThreshold}
              onValueChange={(value) => updateSetting('confidenceThreshold', value)}
              minimumTrackTintColor={theme.colors.accent}
              maximumTrackTintColor={theme.colors.surface}
              thumbTintColor={theme.colors.accent}
            />
          </View>
        </SettingItem>
        <SettingItem
          icon={<RotateCcw />}
          title="Reset Settings"
          description="Reset all settings to default values"
          iconColor={theme.colors.error}
          showArrow
          onPress={handleResetSettings}
        />
      </View>

      {/* About */}
      <View
        style={[styles.sectionCard, {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.colors.primary,
        }]}
      >
        <Text style={[styles.sectionHeader, { color: theme.colors.primary }]}>About</Text>
        <SettingItem
          icon={<Info />}
          title="App Version"
          description="1.0.0 (PyTorch Mobile Optimized)"
          iconColor={theme.colors.textSecondary}
          showArrow
          onPress={() => showAlert('Version Info', 'AudioSense v1.0.0\nBuilt with PyTorch Mobile for on-device inference')}
        />
        <SettingItem
          icon={<Shield />}
          title="Privacy Policy"
          description="Learn about our privacy practices"
          iconColor={theme.colors.info}
          showArrow
          onPress={() => showAlert('Privacy Policy', 'Your privacy is our priority. All audio processing happens locally on your device. No data is transmitted to external servers.')}
        />
      </View>

      {/* Model Information */}
      <View
        style={[styles.sectionCard, {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.colors.primary,
        }]}
      >
        <View style={styles.modelHeader}>
          <Zap size={20} color={theme.colors.accent} />
          <Text style={[styles.modelTitle, { color: theme.colors.text }]}>PyTorch Model Information</Text>
        </View>
        <View style={styles.modelStats}>
          <View style={styles.modelStat}>
            <Text style={[styles.modelStatLabel, { color: theme.colors.textSecondary }]}>Model Size</Text>
            <Text style={[styles.modelStatValue, { color: theme.colors.primary }]}>1.8 MB</Text>
          </View>
          <View style={styles.modelStat}>
            <Text style={[styles.modelStatLabel, { color: theme.colors.textSecondary }]}>Inference Time</Text>
            <Text style={[styles.modelStatValue, { color: theme.colors.primary }]}>~8ms</Text>
          </View>
          <View style={styles.modelStat}>
            <Text style={[styles.modelStatLabel, { color: theme.colors.textSecondary }]}>Supported Events</Text>
            <Text style={[styles.modelStatValue, { color: theme.colors.primary }]}>12 types</Text>
          </View>
        </View>
        <Text style={[styles.modelDescription, { color: theme.colors.textSecondary }]}>Lightweight PyTorch Mobile model optimized for on-device inference. Trained on DCASE dataset with additional real-world samples for improved accuracy.</Text>
      </View>
    </ScrollView>
  );

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: isDark ? '#0f1123' : theme.colors.background }]}>
        {isDark ? (
          <LinearGradient colors={theme.gradients.background} style={styles.gradient}>
            <View style={styles.loadingContainer}>
              <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>Loading settings...</Text>
            </View>
          </LinearGradient>
        ) : (
          <View style={styles.loadingContainer}>
            <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>Loading settings...</Text>
          </View>
        )}
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: isDark ? '#0f1123' : theme.colors.background }]}>
      {isDark ? (
        <LinearGradient colors={theme.gradients.background} style={styles.gradient}>
          {renderContent()}
        </LinearGradient>
      ) : (
        renderContent()
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    fontSize: 16,
  },
  content: {
    padding: 24,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginTop: 8,
  },
  subtitle: {
    fontSize: 16,
    marginTop: 8,
    textAlign: 'center',
  },
  card: {
    borderRadius: 16,
    borderWidth: 1,
    marginBottom: 24,
    overflow: 'hidden', // to clip the divider
  },
  sectionCard: {
    borderRadius: 28,
    backgroundColor: 'transparent', // Let gradient show through
    borderWidth: 2.5,
    shadowOpacity: 0.18,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: 10 },
    padding: 28,
    marginBottom: 28,
    elevation: 12,
  },
  sectionHeader: {
    fontWeight: 'bold',
    fontSize: 18,
    marginBottom: 16,
  },
  settingItem: {
    borderRadius: 12,
    marginBottom: 8,
    borderWidth: 1,
  },
  settingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  settingIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  settingText: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 2,
  },
  settingDescription: {
    fontSize: 14,
  },
  sliderContainer: {
    marginTop: 8,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  modelInfo: {
    borderRadius: 16,
    padding: 20,
    marginBottom: 40,
    borderWidth: 1,
  },
  modelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  modelTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  modelStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  modelStat: {
    alignItems: 'center',
  },
  modelStatLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  modelStatValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  modelDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
});