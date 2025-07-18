import { useState, useEffect, useRef } from 'react';

export interface AnalyticsData {
  totalEvents: number;
  avgConfidence: number;
  detectionAccuracy: number;
  uptime: string;
  topEvents: Array<{ type: string; count: number }>;
  hourlyData: Array<{ hour: string; count: number }>;
  recentActivity: Array<{ timestamp: Date; event: string; confidence: number }>;
  systemHealth: {
    cpuUsage: number;
    memoryUsage: number;
    batteryLevel: number;
    networkStatus: string;
  };
}

export const useRealTimeAnalytics = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [isActive, setIsActive] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<Date>(new Date());

  useEffect(() => {
    if (isActive) {
      startAnalytics();
    } else {
      stopAnalytics();
    }

    return () => stopAnalytics();
  }, [isActive]);

  const startAnalytics = () => {
    startTimeRef.current = new Date();
    updateAnalytics();
    
    intervalRef.current = setInterval(() => {
      updateAnalytics();
    }, 2000); // Update every 2 seconds for real-time feel
  };

  const stopAnalytics = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const updateAnalytics = () => {
    const now = new Date();
    const uptimeMs = now.getTime() - startTimeRef.current.getTime();
    const uptimeHours = Math.floor(uptimeMs / (1000 * 60 * 60));
    const uptimeMinutes = Math.floor((uptimeMs % (1000 * 60 * 60)) / (1000 * 60));

    // Simulate real-time data with some randomness
    const baseEvents = analyticsData?.totalEvents || 0;
    const newEvents = Math.floor(Math.random() * 3); // 0-2 new events
    
    const eventTypes = [
      { type: 'airport' },
      { type: 'bus' },
      { type: 'metro' },
      { type: 'park' },
      { type: 'shopping_mall' },
    ];

    // Generate hourly data for the last 24 hours
    const hourlyData = [];
    for (let i = 23; i >= 0; i--) {
      const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
      const hourStr = hour.getHours().toString().padStart(2, '0');
      const count = Math.floor(Math.random() * 15) + (i < 8 ? Math.random() * 20 : 0);
      hourlyData.push({ hour: hourStr, count });
    }

    // Generate top events with realistic distribution
    const topEvents = eventTypes.map(event => ({
      ...event,
      count: Math.floor(Math.random() * 50) + 10
    })).sort((a, b) => b.count - a.count).slice(0, 4);

    // Generate recent activity
    const recentActivity = [];
    for (let i = 0; i < 10; i++) {
      const timestamp = new Date(now.getTime() - Math.random() * 60 * 60 * 1000);
      const event = eventTypes[Math.floor(Math.random() * eventTypes.length)];
      recentActivity.push({
        timestamp,
        event: event.type,
        confidence: 0.6 + Math.random() * 0.4
      });
    }

    setAnalyticsData({
      totalEvents: baseEvents + newEvents,
      avgConfidence: 0.82 + Math.random() * 0.15,
      detectionAccuracy: 0.89 + Math.random() * 0.08,
      uptime: `${uptimeHours}h ${uptimeMinutes}m`,
      topEvents,
      hourlyData,
      recentActivity: recentActivity.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()),
      systemHealth: {
        cpuUsage: 15 + Math.random() * 25,
        memoryUsage: 45 + Math.random() * 30,
        batteryLevel: 75 + Math.random() * 20,
        networkStatus: Math.random() > 0.1 ? 'Connected' : 'Offline'
      }
    });
  };

  const addEvent = (eventType: string, confidence: number) => {
    if (!analyticsData) return;

    const newActivity = {
      timestamp: new Date(),
      event: eventType,
      confidence
    };

    setAnalyticsData(prev => ({
      ...prev!,
      totalEvents: prev!.totalEvents + 1,
      recentActivity: [newActivity, ...prev!.recentActivity.slice(0, 9)]
    }));
  };

  return {
    analyticsData,
    isActive,
    setIsActive,
    addEvent
  };
};