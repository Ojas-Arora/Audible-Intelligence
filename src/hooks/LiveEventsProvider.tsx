import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export interface LiveEvent {
  type: string;
  confidence: number;
  timestamp: Date;
  category: string;
}

export interface LiveStats {
  totalEvents: number;
  avgConfidence: number;
  detectionAccuracy: number;
  topEvents: Array<{ type: string; count: number }>;
  avgLatency?: number;
}

const EVENT_TYPES = [
  { type: 'airport', category: 'transport' },
  { type: 'bus', category: 'transport' },
  { type: 'metro', category: 'transport' },
  { type: 'park', category: 'public' },
  { type: 'shopping_mall', category: 'public' },
];

interface LiveEventsContextType {
  events: LiveEvent[];
  stats: LiveStats;
  addEvent: (event: LiveEvent) => void;
  getLastDetectedTimes: () => { [key: string]: Date | null };
}

const LiveEventsContext = createContext<LiveEventsContextType | undefined>(undefined);

export const LiveEventsProvider = ({ children }: { children: ReactNode }) => {
  const [events, setEvents] = useState<LiveEvent[]>([]);

  // Calculate average latency between last N events
  let avgLatency: number | undefined = undefined;
  if (events.length > 1) {
    // Sort by timestamp descending (most recent first)
    const sorted = [...events].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    const N = Math.min(10, sorted.length - 1);
    let totalDiff = 0;
    for (let i = 0; i < N; i++) {
      totalDiff += Math.abs(sorted[i].timestamp.getTime() - sorted[i + 1].timestamp.getTime());
    }
    avgLatency = totalDiff / N;
  }

  const stats: LiveStats = {
    totalEvents: events.length,
    avgConfidence: events.length ? events.reduce((a, b) => a + b.confidence, 0) / events.length : 0,
    detectionAccuracy: 0.92, // static for now
    topEvents: EVENT_TYPES.map(e => ({
      type: e.type,
      count: events.filter(ev => ev.type === e.type).length,
    })).sort((a, b) => b.count - a.count).slice(0, 4),
    avgLatency,
  };

  const getLastDetectedTimes = () => {
    const lastTimes: { [key: string]: Date | null } = {};
    EVENT_TYPES.forEach(e => {
      const found = events.find(ev => ev.type === e.type);
      lastTimes[e.type] = found ? found.timestamp : null;
    });
    return lastTimes;
  };

  const addEvent = useCallback((event: LiveEvent) => {
    setEvents(prev => [event, ...prev.slice(0, 49)]);
  }, []);

  return (
    <LiveEventsContext.Provider value={{ events, stats, addEvent, getLastDetectedTimes }}>
      {children}
    </LiveEventsContext.Provider>
  );
};

export const useLiveEvents = () => {
  const context = useContext(LiveEventsContext);
  if (!context) {
    throw new Error('useLiveEvents must be used within a LiveEventsProvider');
  }
  return context;
};
