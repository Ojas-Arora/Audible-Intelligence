import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { View } from 'react-native';
import { FederatedLearning } from '@/components/FederatedLearning';

export default function LearningScreen() {
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#0f1123' }}>
      <LinearGradient
        colors={['#0f1123', '#23255d', '#1a1a40']}
        style={{ flex: 1 }}
      >
        <View style={{ flex: 1, padding: 0 }}>
          <FederatedLearning />
        </View>
      </LinearGradient>
    </SafeAreaView>
  );
}