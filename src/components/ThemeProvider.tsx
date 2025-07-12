import React, { createContext, useContext, useState, useEffect } from 'react';
import { Appearance, ColorSchemeName } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

type GradientColors = readonly [string, string, ...string[]];

export interface Theme {
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    card: string;
    text: string;
    textSecondary: string;
    border: string;
    success: string;
    warning: string;
    error: string;
    info: string;
    // Event type colors
    dogBark: string;
    carHorn: string;
    alarm: string;
    glassBreak: string;
    doorSlam: string;
    siren: string;
    footsteps: string;
    speech: string;
    music: string;
    machinery: string;
    nature: string;
    silence: string;
  };
  gradients: {
    primary: GradientColors;
    secondary: GradientColors;
    accent: GradientColors;
    background: GradientColors;
    hero: GradientColors;
    card: GradientColors;
  };
}

const lightTheme: Theme = {
  colors: {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    accent: '#f59e0b',
    background: '#ffffff',
    surface: '#f8fafc',
    card: '#ffffff',
    text: '#1e293b',
    textSecondary: '#64748b',
    border: '#e2e8f0',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',

    // Event type colors
    dogBark: '#f59e0b',
    carHorn: '#ef4444',
    alarm: '#dc2626',
    glassBreak: '#6366f1',
    doorSlam: '#3b82f6',
    siren: '#f97316',
    footsteps: '#10b981',
    speech: '#06b6d4',
    music: '#a855f7',
    machinery: '#64748b',
    nature: '#059669',
    silence: '#6b7280',
  },
  gradients: {
    primary: ['#6366f1', '#8b5cf6'] as const,
    secondary: ['#8b5cf6', '#a855f7'] as const,
    accent: ['#f59e0b', '#f97316'] as const,
    background: ['#ffffff', '#f8fafc'] as const,
    hero: ['#6366f1', '#8b5cf6', '#a855f7'] as const,
    card: ['#ffffff', '#f8fafc'] as const,
  },
};

const darkTheme: Theme = {
  colors: {
    primary: '#818cf8',
    secondary: '#a78bfa',
    accent: '#fbbf24',
    background: '#0f172a',
    surface: '#1e293b',
    card: '#334155',
    text: '#f1f5f9',
    textSecondary: '#94a3b8',
    border: '#475569',
    success: '#34d399',
    warning: '#fbbf24',
    error: '#f87171',
    info: '#60a5fa',

    // Event type colors
    dogBark: '#fbbf24',
    carHorn: '#f87171',
    alarm: '#ef4444',
    glassBreak: '#818cf8',
    doorSlam: '#60a5fa',
    siren: '#fb923c',
    footsteps: '#34d399',
    speech: '#22d3ee',
    music: '#c084fc',
    machinery: '#94a3b8',
    nature: '#10b981',
    silence: '#9ca3af',
  },
  gradients: {
    primary: ['#0f172a', '#1e293b', '#334155'] as const,
    secondary: ['#1e293b', '#334155', '#475569'] as const,
    accent: ['#fbbf24', '#f59e0b'] as const,
    background: ['#0f172a', '#1e293b'] as const,
    hero: ['#0f172a', '#1e293b', '#334155', '#6366f1'] as const,
    card: ['#334155', '#475569'] as const,
  },
};

interface ThemeContextType {
  theme: Theme;
  isDark: boolean;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Load saved theme preference
    const loadTheme = async () => {
      try {
        const savedTheme = await AsyncStorage.getItem('theme');
        if (savedTheme) {
          setIsDark(savedTheme === 'dark');
        } else {
          // Use system preference
          const systemTheme = Appearance.getColorScheme();
          setIsDark(systemTheme === 'dark');
        }
      } catch (error) {
        console.error('Error loading theme:', error);
      }
    };

    loadTheme();

    // Listen for system theme changes
    const subscription = Appearance.addChangeListener(({ colorScheme }) => {
      AsyncStorage.getItem('theme').then((savedTheme) => {
        if (!savedTheme) {
          setIsDark(colorScheme === 'dark');
        }
      });
    });

    return () => subscription?.remove();
  }, []);

  const toggleTheme = async () => {
    const newTheme = !isDark;
    setIsDark(newTheme);
    try {
      await AsyncStorage.setItem('theme', newTheme ? 'dark' : 'light');
    } catch (error) {
      console.error('Error saving theme:', error);
    }
  };

  const theme = isDark ? darkTheme : lightTheme;

  return (
    <ThemeContext.Provider value={{ theme, isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};