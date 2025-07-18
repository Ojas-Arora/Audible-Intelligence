import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Platform,
  Animated,
  TextInput,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Clock, Filter, Search, MoveVertical as MoreVertical, Calendar, TrendingUp, Activity, ChartBar as BarChart3, Zap, Eye, CircleCheck as CheckCircle, CircleAlert as AlertCircle, Volume2 } from 'lucide-react-native';
import { useLiveEvents } from '@/hooks/useLiveEvents';
import { useTheme } from '@/components/ThemeProvider';

const { width } = Dimensions.get('window');

interface Event {
  id: string;
  type: string;
  confidence: number;
  timestamp: Date;
  duration: number;
  icon: string;
  location?: string;
}

const AnimatedTouchableOpacity = Animated.createAnimatedComponent(TouchableOpacity);

// Emoji icons for event types
import { Plane } from 'lucide-react-native';

const EVENT_TYPE_ICONS: Record<string, React.ReactNode> = {
  airport: <Plane size={24} color="#7B61FF" />,
  bus: <Text style={{fontSize: 24}}>🚌</Text>,
  metro: <Text style={{fontSize: 24}}>🚇</Text>,
  park: <Text style={{fontSize: 24}}>🌳</Text>,
  shopping_mall: <Text style={{fontSize: 24}}>🏬</Text>,
};

const EventCard = ({ event, theme, index }: { event: any; theme: any; index: number }) => {
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
    }, index * 100);
  }, []);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return theme.colors.success;
    if (confidence >= 0.7) return theme.colors.warning;
    return theme.colors.error;
  };

  const getTimeAgo = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  return (
    <AnimatedTouchableOpacity
      style={[
        styles.eventCard,
        {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          transform: [{ scale: scaleAnim }],
          opacity: opacityAnim,
        },
      ]}
      activeOpacity={0.8}
    >
      <LinearGradient
        colors={[theme.colors.card, theme.colors.surface]}
        style={styles.eventCardGradient}
      >
        <View style={styles.eventCardHeader}>
          <View style={styles.eventIconContainer}>
            {EVENT_TYPE_ICONS[event.type] || <Text style={styles.eventIcon}>❓</Text>}
          </View>
          <View style={styles.eventInfo}>
            <Text style={[styles.eventType, { color: theme.colors.text }]}>
              {event.type.replace(/_/g, ' ').toUpperCase()}
            </Text>
            <View style={styles.eventMeta}>
              <Clock size={12} color={theme.colors.textSecondary} />
              <Text style={[styles.eventTime, { color: theme.colors.textSecondary }]}>
                {getTimeAgo(event.timestamp)}
              </Text>
            </View>
          </View>
          <TouchableOpacity style={styles.moreButton}>
            <MoreVertical size={16} color={theme.colors.textSecondary} />
          </TouchableOpacity>
        </View>

        <View style={styles.eventCardBody}>
          <View style={styles.confidenceContainer}>
            <Text style={[styles.confidenceLabel, { color: theme.colors.textSecondary }]}>
              Confidence
            </Text>
            <View style={[styles.confidenceBar, { backgroundColor: theme.colors.surface }]}>
              <View
                style={[
                  styles.confidenceFill,
                  {
                    width: `${event.confidence * 100}%`,
                    backgroundColor: getConfidenceColor(event.confidence),
                  },
                ]}
              />
            </View>
            <Text style={[styles.confidenceValue, { color: theme.colors.text }]}>
              {Math.round(event.confidence * 100)}%
            </Text>
          </View>

          <View style={styles.eventActions}>
            <TouchableOpacity style={[styles.actionButton, { backgroundColor: theme.colors.primary + '20' }]}>
              <Eye size={14} color={theme.colors.primary} />
              <Text style={[styles.actionText, { color: theme.colors.primary }]}>View</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionButton, { backgroundColor: theme.colors.success + '20' }]}>
              <CheckCircle size={14} color={theme.colors.success} />
              <Text style={[styles.actionText, { color: theme.colors.success }]}>Verify</Text>
            </TouchableOpacity>
          </View>
        </View>
      </LinearGradient>
    </AnimatedTouchableOpacity>
  );
};

const StatCard = ({ icon, value, label, color, change, delay = 0 }: any) => {
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
    outputRange: [20, 0],
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
        {React.cloneElement(icon, { size: 18, color })}
      </View>
      <Text style={[styles.statValue, { color: theme.colors.text }]}>{value}</Text>
      <Text style={[styles.statLabel, { color: theme.colors.textSecondary }]}>{label}</Text>
      {change && (
        <View style={styles.statChange}>
          <TrendingUp size={12} color={theme.colors.success} />
          <Text style={[styles.statChangeText, { color: theme.colors.success }]}>{change}</Text>
        </View>
      )}
    </Animated.View>
  );
};

export default function EventsScreen() {
  const { events, stats, getLastDetectedTimes } = useLiveEvents();
  const { theme, isDark } = useTheme();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTimeRange, setSelectedTimeRange] = useState('today');
  const [selectedTypeFilter, setSelectedTypeFilter] = useState('all');

  // Only use the 5 real event types
  const EVENT_TYPES = ['airport', 'bus', 'metro', 'park', 'shopping_mall'];
  const timeRanges = ['today', 'week', 'month', 'all'];

  // Filtering logic for search and time range
  const filteredEvents = events.filter(event => {
    const matchesSearch = event.type.toLowerCase().includes(searchQuery.toLowerCase());
    const now = new Date();
    const eventDate = new Date(event.timestamp);
    let matchesTimeRange = true;
    if (selectedTimeRange === 'today') {
      matchesTimeRange = eventDate.toDateString() === now.toDateString();
    } else if (selectedTimeRange === 'week') {
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      matchesTimeRange = eventDate >= weekAgo;
    } else if (selectedTimeRange === 'month') {
      const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      matchesTimeRange = eventDate >= monthAgo;
    }
    const matchesType = selectedTypeFilter === 'all' ? true : event.type === selectedTypeFilter;
    return matchesSearch && matchesTimeRange && matchesType;
  });

  // Stats per event type
  const lastDetectedTimes = getLastDetectedTimes();
   EVENT_TYPES.map(type => {
    const count = events.filter(e => e.type === type).length;
    const lastTime = lastDetectedTimes[type];
    return {
      type,
      count,
      lastDetected: lastTime ? new Date(lastTime) : null,
    };
  });

  // No more legacy stats - show count and last detected for each event
  // This replaces the old StatCard grid


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
            <BarChart3 size={32} color="white" />
          </View>
          <Text style={styles.heroTitle}>Event Analytics</Text>
          <Text style={styles.heroSubtitle}>
            Real-time insights into detected audio events
          </Text>
        </View>
      </LinearGradient>

      {/* Stats Section */}
      <View style={styles.statsSection}>
        <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Performance Overview</Text>
        <View style={styles.statsGrid}>
          <StatCard
            icon={<Activity />}
            value={stats.totalEvents}
            label="Total Events"
            color="#7B61FF"
            change="+12%"
          />
          <StatCard
            icon={<TrendingUp />}
            value={`${Math.round(stats.avgConfidence * 100)}%`}
            label="Avg Confidence"
            color="#2ECC71"
            change="+5%"
          />
          <StatCard
            icon={<Zap />}
            value={typeof stats.avgLatency === 'number' && stats.avgLatency > 0 ? `${stats.avgLatency.toFixed(1)}ms` : 'N/A'}
            label="Avg Latency"
            color="#F7B731"
            change={stats.avgLatency ? '' : undefined}
          />
          <StatCard
            icon={<Volume2 />}
            value={events.filter(e => e.confidence >= 0.9).length}
            label="High Confidence"
            color="#5C7CFA"
            change="+8%"
          />
        </View>
      </View>

      {/* Event Type Filters */}
      <View style={styles.filtersSection}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filterContent}>
          {[
            { label: 'All', value: 'all' },
            { label: 'Airport', value: 'airport' },
            { label: 'Bus', value: 'bus' },
            { label: 'Metro', value: 'metro' },
            { label: 'Park', value: 'park' },
            { label: 'Shopping Mall', value: 'shopping_mall' },
          ].map(filter => (
            <TouchableOpacity
              key={filter.value}
              style={[
                styles.filterButton,
                {
                  backgroundColor: (selectedTypeFilter === filter.value) ? theme.colors.accent : theme.colors.surface,
                  borderColor: (selectedTypeFilter === filter.value) ? theme.colors.accent : theme.colors.border,
                },
              ]}
              onPress={() => setSelectedTypeFilter(filter.value)}
            >
              <Text style={[styles.filterText, { color: (selectedTypeFilter === filter.value) ? 'white' : theme.colors.text }]}>{filter.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
        {/* Time Filters */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filterContent}>
          {timeRanges.map(range => (
            <TouchableOpacity
              key={range}
              style={[
                styles.timeRangeButton,
                {
                  backgroundColor: selectedTimeRange === range ? theme.colors.accent : theme.colors.surface,
                  borderColor: selectedTimeRange === range ? theme.colors.accent : theme.colors.border,
                },
              ]}
              onPress={() => setSelectedTimeRange(range)}
            >
              <Calendar size={14} color={selectedTimeRange === range ? 'white' : theme.colors.textSecondary} />
              <Text
                style={[
                  styles.timeRangeText,
                  { color: selectedTimeRange === range ? 'white' : theme.colors.text },
                ]}
              >
                {range.charAt(0).toUpperCase() + range.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <View style={[
          styles.searchBar,
          {
            backgroundColor: theme.colors.surface,
            borderWidth: 0,
            borderColor: 'transparent',
            shadowColor: 'transparent',
            elevation: 0,
          },
        ]}>
          <Search size={20} color={theme.colors.textSecondary} style={{ marginRight: 8 }} />
          <TextInput
            style={[
              styles.searchInput,
              {
                color: theme.colors.text,
                paddingVertical: 8,
                borderWidth: 0,
                borderColor: 'transparent',
                outlineWidth: 0,
                outlineColor: 'transparent',
                shadowColor: 'transparent',
                elevation: 0,
              },
            ]}
            placeholder="Search events..."
            placeholderTextColor={theme.colors.textSecondary}
            value={searchQuery}
            onChangeText={setSearchQuery}
            underlineColorAndroid="transparent"
            selectionColor={theme.colors.accent}
          />
        </View>
      </View>

      {/* Events List */}
      <View style={styles.eventsSection}>
        <View style={styles.eventsSectionHeader}>
          <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>Recent Events</Text>
          <Text style={[styles.eventsCount, { color: theme.colors.textSecondary }]}>{filteredEvents.length} events</Text>
        </View>
        {filteredEvents.length === 0 ? (
          <View style={styles.emptyContainer}>
            <AlertCircle size={48} color={theme.colors.textSecondary} />
            <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>No events found</Text>
            <Text style={[styles.emptySubText, { color: theme.colors.textSecondary }]}>Try adjusting your search or filter criteria</Text>
          </View>
        ) : (
          <View style={styles.eventsList}>
            {filteredEvents.map((event, index) => (
              <EventCard
                key={index}
                event={event}
                theme={theme}
                index={index}
              />
            ))}
          </View>
        )}
      </View>
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
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 8,
  },
  heroSubtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    textAlign: 'center',
  },
  statsSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: (width - 60) / 2,
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: 'center',
  },
  statIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 4,
  },
  statChange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  statChangeText: {
    fontSize: 10,
    fontWeight: '600',
  },
  filtersSection: {
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  searchContainer: {
    marginBottom: 16,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    borderWidth: 1,
    gap: 12,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
  },
  filterContent: {
    gap: 8,
    paddingVertical: 8,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
  },
  filterText: {
    fontSize: 14,
    fontWeight: '600',
  },
  timeRangeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    borderWidth: 1,
    gap: 6,
  },
  timeRangeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  eventsSection: {
    paddingHorizontal: 24,
  },
  eventsSectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  eventsCount: {
    fontSize: 14,
  },
  eventsList: {
    gap: 12,
  },
  eventCard: {
    borderRadius: 16,
    borderWidth: 1,
    overflow: 'hidden',
  },
  eventCardGradient: {
    padding: 16,
  },
  eventCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  eventIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  eventIcon: {
    fontSize: 24,
  },
  eventInfo: {
    flex: 1,
  },
  eventType: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  eventMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  eventTime: {
    fontSize: 12,
  },
  moreButton: {
    padding: 4,
  },
  eventCardBody: {
    gap: 16,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  confidenceLabel: {
    fontSize: 12,
    minWidth: 60,
  },
  confidenceBar: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 3,
  },
  confidenceValue: {
    fontSize: 12,
    fontWeight: '600',
    minWidth: 36,
    textAlign: 'right',
  },
  eventActions: {
    flexDirection: 'row',
    gap: 8,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    gap: 4,
  },
  actionText: {
    fontSize: 12,
    fontWeight: '600',
  },
  emptyContainer: {
    alignItems: 'center',
    paddingVertical: 48,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubText: {
    fontSize: 14,
    textAlign: 'center',
  },
});