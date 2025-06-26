import { Tabs } from 'expo-router';
import { Mic, Activity, Settings, ChartBar as BarChart3 } from 'lucide-react-native';
import { View } from 'react-native';
import { useTheme } from '@/components/ThemeProvider';

export default function TabLayout() {
  const { theme } = useTheme();

  return (
    <View style={{ flex: 1, backgroundColor: theme.colors.background }}>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarStyle: {
            backgroundColor: theme.colors.surface,
            borderTopColor: theme.colors.border,
            borderTopWidth: 1,
            paddingBottom: 8,
            paddingTop: 8,
            height: 80,
          },
          tabBarActiveTintColor: theme.colors.primary,
          tabBarInactiveTintColor: theme.colors.textSecondary,
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: '600',
            marginTop: 4,
          },
        }}>
        <Tabs.Screen
          name="dashboard"
          options={{
            title: 'Dashboard',
            tabBarIcon: ({ size, color }) => (
              <BarChart3 size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="index"
          options={{
            title: 'Detection',
            tabBarIcon: ({ size, color }) => (
              <Mic size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="events"
          options={{
            title: 'Events',
            tabBarIcon: ({ size, color }) => (
              <Activity size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="learning"
          options={{
            title: 'Learning',
            tabBarIcon: ({ size, color }) => (
              <Activity size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="settings"
          options={{
            title: 'Settings',
            tabBarIcon: ({ size, color }) => (
              <Settings size={size} color={color} />
            ),
          }}
        />
      </Tabs>
    </View>
  );
}