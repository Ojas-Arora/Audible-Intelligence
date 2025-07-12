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
  Dimensions,
  Modal,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Settings as SettingsIcon, Volume2, Shield, Bell, Mic, Database, Info, ChevronRight, TriangleAlert as AlertTriangle, Zap, Moon, Sun, Smartphone, RotateCcw, X, Lock, Eye, FileSliders as Sliders, CircleHelp as HelpCircle } from 'lucide-react-native';
import { useTheme } from '@/components/ThemeProvider';
import { useAppSettings } from '@/hooks/useAppSettings';

const { width, height } = Dimensions.get('window');

// Cross-platform Slider Component
const CrossPlatformSlider = (props: any) => {
  const { theme } = useTheme();
  
  if (Platform.OS === 'web') {
    return (
      <input
        type="range"
        min={props.minimumValue}
        max={props.maximumValue}
        step={0.01}
        value={props.value}
        onChange={(e) => props.onValueChange(parseFloat(e.target.value))}
        style={{
          width: '100%',
          height: '6px',
          borderRadius: '3px',
          background: `linear-gradient(to right, ${props.minimumTrackTintColor} 0%, ${props.minimumTrackTintColor} ${(props.value - props.minimumValue) / (props.maximumValue - props.minimumValue) * 100}%, ${props.maximumTrackTintColor} ${(props.value - props.minimumValue) / (props.maximumValue - props.minimumValue) * 100}%, ${props.maximumTrackTintColor} 100%)`,
          outline: 'none',
          appearance: 'none',
          cursor: 'pointer',
        }}
      />
    );
  }

  // For mobile, use a custom slider implementation
  const [isDragging, setIsDragging] = useState(false);
  
  const handleTouch = (event: any) => {
    const { locationX } = event.nativeEvent;
    const sliderWidth = 200; // Approximate slider width
    const percentage = Math.max(0, Math.min(1, locationX / sliderWidth));
    const newValue = props.minimumValue + percentage * (props.maximumValue - props.minimumValue);
    props.onValueChange(newValue);
  };

  const progressPercentage = ((props.value - props.minimumValue) / (props.maximumValue - props.minimumValue)) * 100;

  return (
    <View style={styles.customSliderContainer}>
      <TouchableOpacity
        style={[styles.customSliderTrack, { backgroundColor: props.maximumTrackTintColor }]}
        onPress={handleTouch}
        activeOpacity={1}
      >
        <View 
          style={[
            styles.customSliderProgress, 
            { 
              backgroundColor: props.minimumTrackTintColor,
              width: `${progressPercentage}%`
            }
          ]} 
        />
        <View 
          style={[
            styles.customSliderThumb, 
            { 
              backgroundColor: props.thumbTintColor || theme.colors.primary,
              left: `${progressPercentage}%`,
              marginLeft: -8
            }
          ]} 
        />
      </TouchableOpacity>
    </View>
  );
};

export default function SettingsScreen() {
  const { theme, isDark, toggleTheme } = useTheme();
  const { settings, updateSetting, resetSettings, isLoading } = useAppSettings();
  const [activeModal, setActiveModal] = useState<string | null>(null);

  const showAlert = (title: string, message: string) => {
    if (Platform.OS === 'web') {
      alert(`${title}\n\n${message}`);
    } else {
      Alert.alert(title, message);
    }
  };

  const handleResetSettings = () => {
    const confirmReset = () => {
      resetSettings();
      showAlert('Settings Reset', 'All settings have been reset to default values.');
    };

    if (Platform.OS === 'web') {
      if (confirm('Are you sure you want to reset all settings to default values?')) {
        confirmReset();
      }
    } else {
      Alert.alert(
        'Reset Settings',
        'Are you sure you want to reset all settings to default values?',
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Reset', style: 'destructive', onPress: confirmReset }
        ]
      );
    }
  };

  const SettingCard = ({ 
    icon, 
    title, 
    description, 
    children,
    onPress,
    showArrow = false,
    gradient = false
  }: {
    icon: React.ReactElement;
    title: string;
    description?: string;
    children?: React.ReactNode;
    onPress?: () => void;
    showArrow?: boolean;
    gradient?: boolean;
  }) => (
    <TouchableOpacity 
      style={[
        styles.settingCard,
        { 
          backgroundColor: gradient ? 'transparent' : theme.colors.card,
          borderColor: theme.colors.border 
        }
      ]}
      onPress={onPress}
      activeOpacity={onPress ? 0.7 : 1}
    >
      {gradient ? (
        <LinearGradient
          colors={theme.gradients.card}
          style={styles.cardGradient}
        >
          <SettingCardContent 
            icon={icon}
            title={title}
            description={description}
            showArrow={showArrow}
            children={children}
          />
        </LinearGradient>
      ) : (
        <SettingCardContent 
          icon={icon}
          title={title}
          description={description}
          showArrow={showArrow}
          children={children}
        />
      )}
    </TouchableOpacity>
  );

  const SettingCardContent = ({ icon, title, description, showArrow, children }: any) => (
    <View style={styles.settingContent}>
      <View style={[styles.settingIcon, { backgroundColor: theme.colors.primary + '20' }]}>
        {React.cloneElement(icon, { size: 24, color: theme.colors.primary })}
      </View>
      
      <View style={styles.settingText}>
        <Text style={[styles.settingTitle, { color: theme.colors.text }]}>{title}</Text>
        {description && (
          <Text style={[styles.settingDescription, { color: theme.colors.textSecondary }]}>
            {description}
          </Text>
        )}
        {children}
      </View>

      {showArrow && (
        <ChevronRight size={20} color={theme.colors.textSecondary} />
      )}
    </View>
  );

  const SliderSetting = ({ 
    icon, 
    title, 
    value, 
    onValueChange, 
    minimumValue = 0, 
    maximumValue = 1,
    formatValue = (v: number) => `${Math.round(v * 100)}%`
  }: {
    icon: React.ReactElement;
    title: string;
    value: number;
    onValueChange: (value: number) => void;
    minimumValue?: number;
    maximumValue?: number;
    formatValue?: (value: number) => string;
  }) => (
    <SettingCard
      icon={icon}
      title={title}
      description={`Current: ${formatValue(value)}`}
    >
      <View style={styles.sliderContainer}>
        <CrossPlatformSlider
          style={styles.slider}
          minimumValue={minimumValue}
          maximumValue={maximumValue}
          value={value}
          onValueChange={onValueChange}
          minimumTrackTintColor={theme.colors.primary}
          maximumTrackTintColor={theme.colors.border}
          thumbTintColor={theme.colors.primary}
        />
        <View style={styles.sliderLabels}>
          <Text style={[styles.sliderLabel, { color: theme.colors.textSecondary }]}>
            {formatValue(minimumValue)}
          </Text>
          <Text style={[styles.sliderLabel, { color: theme.colors.textSecondary }]}>
            {formatValue(maximumValue)}
          </Text>
        </View>
      </View>
    </SettingCard>
  );

  const SwitchSetting = ({ 
    icon, 
    title, 
    description, 
    value, 
    onValueChange,
    gradient = false
  }: {
    icon: React.ReactElement;
    title: string;
    description?: string;
    value: boolean;
    onValueChange: (value: boolean) => void;
    gradient?: boolean;
  }) => (
    <SettingCard
      icon={icon}
      title={title}
      description={description}
      gradient={gradient}
    >
      <Switch
        value={value}
        onValueChange={onValueChange}
        thumbColor={value ? theme.colors.primary : theme.colors.surface}
        trackColor={{ false: theme.colors.border, true: theme.colors.primary + '33' }}
        style={styles.switch}
      />
    </SettingCard>
  );

  const InfoModal = ({ title, content, visible, onClose }: {
    title: string;
    content: string;
    visible: boolean;
    onClose: () => void;
  }) => (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onClose}
    >
      <View style={styles.modalOverlay}>
        <View style={[styles.modalContent, { backgroundColor: theme.colors.card }]}>
          <View style={styles.modalHeader}>
            <Text style={[styles.modalTitle, { color: theme.colors.text }]}>{title}</Text>
            <TouchableOpacity onPress={onClose} style={styles.modalClose}>
              <X size={24} color={theme.colors.textSecondary} />
            </TouchableOpacity>
          </View>
          <Text style={[styles.modalText, { color: theme.colors.textSecondary }]}>{content}</Text>
          <TouchableOpacity 
            style={[styles.modalButton, { backgroundColor: theme.colors.primary }]}
            onPress={onClose}
          >
            <Text style={styles.modalButtonText}>Got it</Text>
          </TouchableOpacity>
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
            <SettingsIcon size={32} color="white" />
          </View>
          <Text style={styles.heroTitle}>Settings</Text>
          <Text style={styles.heroSubtitle}>
            Customize your audio detection experience
          </Text>
        </View>
      </LinearGradient>

      {/* Quick Actions */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Quick Actions</Text>
        <View style={styles.quickActions}>
          <SwitchSetting
            icon={isDark ? <Sun /> : <Moon />}
            title="Dark Mode"
            description="Switch between light and dark themes"
            value={isDark}
            onValueChange={toggleTheme}
            gradient
          />
          
          <SwitchSetting
            icon={<Bell />}
            title="Notifications"
            description="Get notified when events are detected"
            value={settings.notifications}
            onValueChange={(v) => updateSetting('notifications', v)}
          />
        </View>
      </View>

      {/* Detection Settings */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Detection</Text>
        
        <SwitchSetting
          icon={<Mic />}
          title="Auto Detection"
          description="Automatically start detection when app opens"
          value={settings.autoDetection}
          onValueChange={(v) => updateSetting('autoDetection', v)}
        />

        <SliderSetting
          icon={<Volume2 />}
          title="Sensitivity Level"
          value={settings.sensitivity}
          onValueChange={(v) => updateSetting('sensitivity', v)}
          minimumValue={0.1}
          maximumValue={1.0}
        />

        <SliderSetting
          icon={<Zap />}
          title="Confidence Threshold"
          value={settings.confidenceThreshold}
          onValueChange={(v) => updateSetting('confidenceThreshold', v)}
          minimumValue={0.3}
          maximumValue={0.9}
        />

        <SwitchSetting
          icon={<Smartphone />}
          title="Background Processing"
          description="Continue detection when app is in background"
          value={settings.backgroundProcessing}
          onValueChange={(v) => updateSetting('backgroundProcessing', v)}
        />
      </View>

      {/* Privacy & Security */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Privacy & Security</Text>
        
        <SettingCard
          icon={<Shield />}
          title="Privacy Policy"
          description="Learn about our privacy practices"
          onPress={() => setActiveModal('privacy')}
          showArrow
        />

        <SwitchSetting
          icon={<Database />}
          title="Save Detections"
          description="Store detection history locally"
          value={settings.saveDetections}
          onValueChange={(v) => updateSetting('saveDetections', v)}
        />

        <SwitchSetting
          icon={<Lock />}
          title="Data Collection"
          description="Help improve detection accuracy (anonymous)"
          value={settings.dataCollection}
          onValueChange={(v) => updateSetting('dataCollection', v)}
        />

        <SettingCard
          icon={<Eye />}
          title="On-Device Processing"
          description="All audio processing happens locally"
          onPress={() => setActiveModal('processing')}
          showArrow
        />
      </View>

      {/* Advanced */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Advanced</Text>
        
        <SwitchSetting
          icon={<Volume2 />}
          title="Haptic Feedback"
          description="Vibrate on event detection (mobile only)"
          value={settings.hapticFeedback}
          onValueChange={(v) => updateSetting('hapticFeedback', v)}
        />

        <SettingCard
          icon={<RotateCcw />}
          title="Reset Settings"
          description="Reset all settings to default values"
          onPress={handleResetSettings}
          showArrow
        />
      </View>

      {/* About */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>About</Text>
        
        <SettingCard
          icon={<Info />}
          title="App Version"
          description="1.0.0 (PyTorch Mobile Optimized)"
          onPress={() => setActiveModal('version')}
          showArrow
        />

        <SettingCard
          icon={<HelpCircle />}
          title="Help & Support"
          description="Get help with using the app"
          onPress={() => setActiveModal('help')}
          showArrow
        />
      </View>

      {/* Model Information */}
      <LinearGradient
        colors={theme.gradients.card}
        style={styles.modelCard}
      >
        <View style={styles.modelHeader}>
          <Zap size={24} color={theme.colors.primary} />
          <Text style={[styles.modelTitle, { color: theme.colors.text }]}>AI Model Info</Text>
        </View>
        <View style={styles.modelStats}>
          <View style={styles.modelStat}>
            <Text style={[styles.modelStatValue, { color: theme.colors.primary }]}>1.8 MB</Text>
            <Text style={[styles.modelStatLabel, { color: theme.colors.textSecondary }]}>Model Size</Text>
          </View>
          <View style={styles.modelStat}>
            <Text style={[styles.modelStatValue, { color: theme.colors.primary }]}>~8ms</Text>
            <Text style={[styles.modelStatLabel, { color: theme.colors.textSecondary }]}>Inference</Text>
          </View>
          <View style={styles.modelStat}>
            <Text style={[styles.modelStatValue, { color: theme.colors.primary }]}>12</Text>
            <Text style={[styles.modelStatLabel, { color: theme.colors.textSecondary }]}>Event Types</Text>
          </View>
        </View>
      </LinearGradient>

      {/* Modals */}
      <InfoModal
        title="Privacy Policy"
        content="Your privacy is our priority. All audio processing happens locally on your device. No data is transmitted to external servers. We collect anonymous usage statistics to improve the app, but you can opt out at any time."
        visible={activeModal === 'privacy'}
        onClose={() => setActiveModal(null)}
      />

      <InfoModal
        title="On-Device Processing"
        content="All audio analysis is performed locally on your device using optimized AI models. This ensures your audio data never leaves your device, providing maximum privacy and security while enabling real-time detection."
        visible={activeModal === 'processing'}
        onClose={() => setActiveModal(null)}
      />

      <InfoModal
        title="Version Information"
        content="AudioSense v1.0.0\n\nBuilt with PyTorch Mobile for on-device inference. Features include real-time audio detection, privacy-preserving processing, and federated learning capabilities."
        visible={activeModal === 'version'}
        onClose={() => setActiveModal(null)}
      />

      <InfoModal
        title="Help & Support"
        content="Need help? Here are some quick tips:\n\n• Adjust sensitivity for better detection\n• Enable auto-detection for convenience\n• Check privacy settings for data control\n• Use background processing for continuous monitoring\n\nFor more help, contact support through the app store."
        visible={activeModal === 'help'}
        onClose={() => setActiveModal(null)}
      />
    </ScrollView>
  );

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <View style={styles.loadingContainer}>
          <Text style={[styles.loadingText, { color: theme.colors.textSecondary }]}>Loading settings...</Text>
        </View>
      </SafeAreaView>
    );
  }

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
  contentContainer: {
    paddingBottom: 32,
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    fontSize: 16,
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
    fontSize: 32,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
  },
  section: {
    marginBottom: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  quickActions: {
    gap: 12,
  },
  settingCard: {
    borderRadius: 16,
    marginBottom: 12,
    borderWidth: 1,
    overflow: 'hidden',
  },
  cardGradient: {
    flex: 1,
  },
  settingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
  },
  settingIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  settingText: {
    flex: 1,
  },
  settingTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
  switch: {
    transform: [{ scaleX: 1.1 }, { scaleY: 1.1 }],
  },
  sliderContainer: {
    marginTop: 16,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  customSliderContainer: {
    width: '100%',
    height: 40,
    justifyContent: 'center',
  },
  customSliderTrack: {
    height: 6,
    borderRadius: 3,
    position: 'relative',
  },
  customSliderProgress: {
    height: '100%',
    borderRadius: 3,
    position: 'absolute',
    left: 0,
    top: 0,
  },
  customSliderThumb: {
    width: 16,
    height: 16,
    borderRadius: 8,
    position: 'absolute',
    top: -5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  sliderLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  sliderLabel: {
    fontSize: 12,
  },
  modelCard: {
    marginHorizontal: 24,
    borderRadius: 20,
    padding: 24,
    marginBottom: 24,
  },
  modelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    gap: 12,
  },
  modelTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  modelStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  modelStat: {
    alignItems: 'center',
  },
  modelStatValue: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  modelStatLabel: {
    fontSize: 12,
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  modalClose: {
    padding: 4,
  },
  modalText: {
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 24,
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
});