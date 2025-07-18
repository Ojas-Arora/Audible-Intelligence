import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, FlatList, KeyboardAvoidingView, Platform } from 'react-native';
import { MessageCircle, X, Send } from 'lucide-react-native';
import { useTheme } from './ThemeProvider';

const FAQ = [
  {
    q: 'What does this app do?',
    a: 'This app provides privacy-first, on-device acoustic detection and smart environment features. It detects events, analyzes audio locally, and helps you manage your environment.'
  },
  {
    q: 'How is my privacy protected?',
    a: 'All audio processing happens locally on your device. No audio is uploaded to the cloud, ensuring your privacy is maintained.'
  },
  {
    q: 'How do I use event detection?',
    a: 'Simply start the app and allow microphone access. The dashboard will show detected events in real time.'
  },
  {
    q: 'What technologies does this app use?',
    a: 'Built with React Native (Expo), expo-av for audio, Animated API for UI, and lucide-react-native icons.'
  },
  {
    q: 'How do I access analytics?',
    a: 'Go to the Analytics tab from the bottom navigation to view detailed stats and trends.'
  },
  {
    q: 'How do I change settings?',
    a: 'Tap the Settings tab to customize app behavior, notifications, and more.'
  },
];

type FAQItem = { q: string; a: string };

const Chatbot = () => {
  const { theme } = useTheme();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hi! I can help explain what this app does and how to use it. Select a question below:' }
  ]);

  const handleFAQPress = (faq: FAQItem) => {
    setMessages((msgs) => [...msgs, { from: 'user', text: faq.q }, { from: 'bot', text: faq.a }]);
  };


  return (
    <>
      {/* Floating Chat Icon */}
      {!open && (
        <TouchableOpacity
          style={[styles.fab, { backgroundColor: theme.colors.primary, shadowColor: theme.colors.primary }]}
          onPress={() => setOpen(true)}
          activeOpacity={0.8}
        >
          <MessageCircle color={theme.colors.background} size={28} />
        </TouchableOpacity>
      )}
      {/* Chat Window */}
      {open && (
        <View style={[styles.chatWindow, { backgroundColor: theme.colors.background, borderColor: theme.colors.primary, shadowColor: theme.colors.primary }]}> 
          <View style={[styles.header, { backgroundColor: theme.colors.primary }]}> 
            <Text style={[styles.headerText, { color: theme.colors.background }]}>Help & Info</Text>
            <TouchableOpacity onPress={() => setOpen(false)}>
              <X color={theme.colors.background} size={22} />
            </TouchableOpacity>
          </View>
          <FlatList
            data={messages}
            keyExtractor={(_, idx) => idx.toString()}
            renderItem={({ item }) => (
              <View style={item.from === 'bot' ? styles.botMsg : styles.userMsg}>
                <Text style={{ color: theme.colors.text }}>{item.text}</Text>
              </View>
            )}
            contentContainerStyle={{ padding: 12 }}
          />
          <View style={styles.faqList}>
            {FAQ.map((faq, i) => (
              <TouchableOpacity
                key={faq.q}
                style={[styles.faqBtn, { backgroundColor: theme.colors.primary + '20', borderColor: theme.colors.primary }]}
                onPress={() => handleFAQPress(faq)}
              >
                <Text style={{ color: theme.colors.primary, fontWeight: '600' }}>{faq.q}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}
    </>
  );
};

const styles = StyleSheet.create({
  fab: {
    position: 'absolute',
    right: 22,
    bottom: 32,
    zIndex: 9999,
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 8,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  chatWindow: {
    position: 'absolute',
    right: 14,
    bottom: 90,
    width: 350,
    maxHeight: 600,
    borderRadius: 20,
    borderWidth: 2,
    zIndex: 99999,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.25,
    shadowRadius: 16,
    overflow: 'hidden',
    elevation: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 14,
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
  },
  headerText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  botMsg: {
    alignSelf: 'flex-start',
    backgroundColor: '#eee8',
    marginVertical: 4,
    padding: 10,
    borderRadius: 10,
    maxWidth: '85%',
  },
  userMsg: {
    alignSelf: 'flex-end',
    backgroundColor: '#cce5ff88',
    marginVertical: 4,
    padding: 10,
    borderRadius: 10,
    maxWidth: '85%',
  },
  faqList: {
    flexDirection: 'column',
    padding: 8,
    gap: 6,
  },
  faqBtn: {
    borderRadius: 8,
    borderWidth: 1.5,
    marginVertical: 3,
    padding: 8,
    marginHorizontal: 2,
  },
});

export default Chatbot;
