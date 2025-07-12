import { useState, useCallback } from 'react';

export interface LiveEvent {
  type: string;
  confidence: number;
  timestamp: Date;
  icon: string;
  category: string;
  processing_time_ms?: number;
}

export interface LiveStats {
  totalEvents: number;
  avgConfidence: number;
  detectionAccuracy: number;
  topEvents: Array<{ type: string; count: number; icon: string }>;
}

const EVENT_TYPES = [
  { type: 'dog_bark', icon: '🐕', category: 'animals' },
  { type: 'car_horn', icon: '🚗', category: 'vehicles' },
  { type: 'alarm', icon: '🚨', category: 'alarms' },
  { type: 'glass_break', icon: '🥃', category: 'home' },
  { type: 'door_slam', icon: '🚪', category: 'home' },
  { type: 'siren', icon: '🚑', category: 'vehicles' },
  { type: 'footsteps', icon: '👣', category: 'home' },
  { type: 'speech', icon: '🗣️', category: 'home' },
];

export function useLiveEvents() {
  const [events, setEvents] = useState<LiveEvent[]>([]);

  // Generate stats from events
  const stats: LiveStats = {
    totalEvents: events.length,
    avgConfidence: events.length ? events.reduce((a, b) => a + b.confidence, 0) / events.length : 0,
    detectionAccuracy: 0.92, // static for now
    topEvents: EVENT_TYPES.map(e => ({
      type: e.type,
      icon: e.icon,
      count: events.filter(ev => ev.type === e.type).length,
    })).sort((a, b) => b.count - a.count).slice(0, 4),
  };

  // Add a new event (to be called from model output)
  const addEvent = useCallback((event: LiveEvent) => {
    setEvents(prev => [event, ...prev.slice(0, 49)]); // keep last 50
  }, []);

  return {
    events,
    stats,
    addEvent,
  };
} 