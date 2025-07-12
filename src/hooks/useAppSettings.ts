import { useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface AppSettings {
  autoDetection: boolean;
  hapticFeedback: boolean;
  dataCollection: boolean;
  darkMode: boolean;
  notifications: boolean;
  highSensitivity: boolean;
  backgroundProcessing: boolean;
  saveDetections: boolean;
  sensitivity: number;
  confidenceThreshold: number;
}

const defaultSettings: AppSettings = {
  autoDetection: false, // Changed to false by default
  hapticFeedback: true,
  dataCollection: false,
  darkMode: false,
  notifications: true, // Notifications enabled by default
  highSensitivity: false,
  backgroundProcessing: true,
  saveDetections: true,
  sensitivity: 0.7,
  confidenceThreshold: 0.6,
};

export const useAppSettings = () => {
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const stored = await AsyncStorage.getItem('appSettings');
      if (stored) {
        const parsedSettings = JSON.parse(stored);
        setSettings({ ...defaultSettings, ...parsedSettings });
      }
    } catch (error) {
      console.error('Error loading settings:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const updateSetting = async <K extends keyof AppSettings>(
    key: K,
    value: AppSettings[K]
  ) => {
    try {
      const newSettings = { ...settings, [key]: value };
      setSettings(newSettings);
      await AsyncStorage.setItem('appSettings', JSON.stringify(newSettings));
    } catch (error) {
      console.error('Error saving setting:', error);
    }
  };

  const resetSettings = async () => {
    try {
      setSettings(defaultSettings);
      await AsyncStorage.setItem('appSettings', JSON.stringify(defaultSettings));
    } catch (error) {
      console.error('Error resetting settings:', error);
    }
  };

  return {
    settings,
    updateSetting,
    resetSettings,
    isLoading,
  };
};