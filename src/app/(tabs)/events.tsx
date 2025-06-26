import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Clock, Filter, Search, MoveVertical as MoreVertical, Calendar, TrendingUp } from 'lucide-react-native';
import { useLiveEvents } from '@/hooks/useLiveEvents';
import { useTheme } from '@/components/ThemeProvider';
import { TextInput } from 'react-native-gesture-handler';

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

export default function EventsScreen() {
  const { events } = useLiveEvents();
  const { theme, isDark } = useTheme();
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [pressedCard, setPressedCard] = useState<string | null>(null);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

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

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return '#10b981';
    if (confidence >= 0.7) return '#f59e0b';
    return '#ef4444';
  };

  const filterTypes = ['all', 'alarms', 'animals', 'vehicles', 'home'];

  const filteredEvents = events.filter(event => {
    const matchesFilter = filter === 'all' || event.type === filter;
    const matchesSearch = event.type.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const renderContent = () => (
    <>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: theme.colors.border }]}>
        <Text style={[styles.title, { color: theme.colors.text }]}>Event History</Text>
        <View style={styles.headerStats}>
          <View style={styles.statItem}>
            <TrendingUp size={16} color={theme.colors.primary} />
            <Text style={[styles.statText, { color: theme.colors.textSecondary }]}>{events.length} events</Text>
          </View>
          <View style={styles.statItem}>
            <Calendar size={16} color={theme.colors.primary} />
            <Text style={[styles.statText, { color: theme.colors.textSecondary }]}>Today</Text>
          </View>
        </View>
      </View>

      {/* Filter and Search Bar */}
      <View style={[styles.filterContainer, { borderBottomColor: theme.colors.border }]}>
        <ScrollView 
          horizontal 
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.filterContent}
        >
          {filterTypes.map((type) => (
            <TouchableOpacity
              key={type}
              style={[
                styles.filterButton,
                { backgroundColor: filter === type ? theme.colors.primary : theme.colors.card,
                  borderColor: filter === type ? theme.colors.primary : theme.colors.border
                }
              ]}
              onPress={() => setFilter(type)}
            >
              <Text style={[
                styles.filterText, 
                { color: filter === type ? theme.colors.card : theme.colors.text }
              ]}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <View style={styles.searchBarContainer}>
        <Search size={20} color={theme.colors.textSecondary} style={styles.searchIcon} />
        <TextInput
          style={[styles.searchInput, { 
            color: theme.colors.text, 
            backgroundColor: theme.colors.surface, 
            borderColor: theme.colors.border 
          }]}
          placeholder="Search events by type..."
          placeholderTextColor={theme.colors.textSecondary}
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
      </View>

      {/* Events List */}
      <ScrollView 
        style={styles.eventsList}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingBottom: 24 }}
      >
        {filteredEvents.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={[styles.emptyText, { color: theme.colors.textSecondary }]}>
              No events found.
            </Text>
            <Text style={[styles.emptySubText, { color: theme.colors.textSecondary }]}>
              Try adjusting your search or filter.
            </Text>
          </View>
        ) : (
          filteredEvents.map((event) => (
            <View
              key={event.timestamp.toString()}
              style={[
                styles.eventCard,
                { 
                  backgroundColor: theme.colors.card,
                  borderColor: theme.colors.border,
                  shadowColor: theme.colors.primary,
                }
              ]}
            >
              <View style={styles.eventCardHeader}>
                <Text style={styles.eventIcon}>{event.icon}</Text>
                <View style={{ flex: 1 }}>
                  <Text style={[styles.eventType, { color: theme.colors.text }]}>{event.type.replace(/_/g, ' ').toUpperCase()}</Text>
                  <View style={styles.eventMeta}>
                    <Clock size={14} color={theme.colors.textSecondary} />
                    <Text style={[styles.eventTime, { color: theme.colors.textSecondary }]}>{getTimeAgo(event.timestamp)}</Text>
                  </View>
                </View>
                <MoreVertical size={20} color={theme.colors.textSecondary} />
              </View>
              <View style={styles.eventCardBody}>
                <View style={{ flex: 1 }}>
                  <Text style={[styles.confidenceLabel, { color: theme.colors.textSecondary }]}>Confidence</Text>
                  <View style={[styles.confidenceBar, { backgroundColor: theme.colors.surface }]}>
                    <View style={[styles.confidenceFill, {
                      width: `${event.confidence * 100}%`,
                      backgroundColor: event.confidence >= 0.9 ? theme.colors.success : event.confidence >= 0.7 ? theme.colors.accent : theme.colors.error,
                    }]} />
                  </View>
                </View>
                <Text style={[styles.confidenceValue, { color: theme.colors.text }]}>{Math.round(event.confidence * 100)}%</Text>
              </View>
            </View>
          ))
        )}
      </ScrollView>
    </>
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: isDark ? '#0f1123' : theme.colors.background }}>
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
  header: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 12,
  },
  headerStats: {
    flexDirection: 'row',
    gap: 24,
  },
  statItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statText: {
    color: '#94a3b8',
    fontSize: 14,
    fontWeight: '600',
  },
  filterContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 20,
    gap: 12,
  },
  filterContent: {
    gap: 8,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#374151',
  },
  filterButtonActive: {
    backgroundColor: '#06b6d4',
  },
  filterText: {
    color: '#94a3b8',
    fontSize: 14,
    fontWeight: '600',
  },
  filterTextActive: {
    color: 'white',
  },
  searchButton: {
    padding: 8,
  },
  eventsList: {
    flex: 1,
    paddingHorizontal: 20,
  },
  eventCard: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  eventCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  eventIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#374151',
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
    color: 'white',
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
  metaSeparator: {
    fontSize: 12,
    color: '#64748b',
  },
  eventLocation: {
    fontSize: 12,
    color: '#64748b',
  },
  moreButton: {
    padding: 4,
  },
  eventDetails: {
    gap: 8,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  confidenceLabel: {
    fontSize: 12,
    marginBottom: 4,
  },
  confidenceBar: {
    flex: 1,
    height: 6,
    backgroundColor: '#374151',
    borderRadius: 3,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 3,
  },
  confidenceValue: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
    minWidth: 36,
    textAlign: 'right',
  },
  durationContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  durationLabel: {
    color: '#94a3b8',
    fontSize: 12,
  },
  durationValue: {
    color: '#06b6d4',
    fontSize: 12,
    fontWeight: '600',
  },
  noEventsMessage: {
    color: '#94a3b8',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 20,
  },
  searchBarContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
  },
  searchIcon: {
    marginRight: 10,
  },
  searchInput: {
    flex: 1,
    padding: 10,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  emptySubText: {
    fontSize: 14,
    color: '#94a3b8',
  },
  eventCardBody: {
    marginTop: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
});