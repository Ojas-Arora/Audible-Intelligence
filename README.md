# AISOC 2025

# Team Name: Audible Intelligence

# 👥 Team Members
- Member 1: Ojas Arora (ojas.arora14@gmail.com)
- Member 2: Diksha (diksha160404@gmail.com)
- Member 3: Chitwan (chitwan.gujrani@gmail.com)

# 🧠 Project Title

Acoustic Source and Event Detection: DCASE Challenge-Inspired Smart App

# 🧩 Assigned Problem Statement

**Project 2: A privacy-first mobile/web application that detects and identifies environmental sounds such as alarms and sirens in real time.**

Traditional acoustic event detection software relies significantly on cloud processing, which results in privacy threats, latency, and poor offline performance. An urgent need exists for an end-to-end, privacy-centered solution that can detect alarms, sirens, or barks accurately in real-time directly on-device—providing low latency, improved security, and smooth performance in low-resource or offline settings.
Alarms, sirens, barking dogs, and other acoustic events frequently report emergencies or require urgent attention. But current sound detection systems have several basic issues that make them difficult to use in everyday life:

- **They don't work without the internet:** Most apps send sound data to servers for processing, so they stop working in offline or low-network areas.

- **They're often slow:** Sending data to the cloud and waiting for a response can cause delays—dangerous in emergencies.

- **They are too heavy for normal phones:** Some apps use complex models that take too much battery, memory, or time to work on regular devices.


To overcome these challenges, our project offers a privacy-oriented, low-latency, and on-device solution for real-time acoustic event detection. We want to solve these problems by building a simple, lightweight mobile/web app that can:

- Detect common and important sounds (like airport, Metro, and Bus)

- Operates completely offline or under low-resource settings

- Be fast, so users get alerts in real-time

- Supports real-world deployment on smartphones, IoT devices, or wearables

- Reliably detects important environmental sounds with high accuracy

This app can help in many real situations—like helping people with hearing difficulties, creating smart home systems, or improving safety during emergencies. This project is motivated by the DCASE Challenge, which aims at practical acoustic scene and event detection tasks. In bringing together the latest AI into a mobile/web app, our aim is to make smarter environments possible—on use cases such as home security, elderly care, public safety systems, and accessibility devices for the hearing-impaired.

## 🚀 Quick Start Guide

This guide will help you set up and run the Audible Intelligence app for both web and mobile environments.

### 1. Prerequisites

#### For Running the App (Web & Mobile):
- **Node.js** (v18 or above recommended)
- **npm** (comes with Node.js)
- **Expo CLI** (for local development & running on devices):
  ```bash
  npm install -g expo-cli
  ```
- **Python** (3.8 or above recommended)
- **pip** (Python package manager)

#### For Backend (Python) Setup:
a) Navigate to the backend directory:
   ```bash
   cd src/send
   ```
b) Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   This will install all necessary dependencies including:
   - numpy
   - tensorflow
   - soundfile
   - librosa
   - matplotlib
   - opencv-python

### 2. Clone the Repository
```bash
git clone <your-fork-url>
cd src
cd Audible-Intelligence
```

### 3. Install Dependencies
```bash
npm install
# or
yarn install
```

### 4. Running the App (Requires Two Terminals)

To enable real-time audio detection, you need to run both the frontend (Expo app) and the backend prediction API simultaneously.

#### Terminal 1: Start the Frontend (Web or Mobile)
#### a) For Web:
  ```bash
  npx expo start -c

   ```
- Open the browser link provided in the terminal by typing w in the terminal.

#### b) For Android/iOS (Mobile):
1. **Install Expo Go** on your mobile device:
   - [Android (Google Play Store)](https://play.google.com/store/apps/details?id=host.exp.exponent)
   - [iOS (App Store)](https://apps.apple.com/app/expo-go/id982107779)

2. **Find your computer's IPv4 address**:
   - Windows: Open Command Prompt and type `ipconfig`
   - Mac/Linux: Open Terminal and type `ifconfig | grep "inet " | grep -v 127.0.0.1`
   - Look for your local IP address (typically starts with 192.168.x.x or 10.0.x.x)

3. **Set the development server IP** (replace 192.168.x.x with your actual IP):
   ```bash
   # On Windows (PowerShell):
   $env:REACT_NATIVE_PACKAGER_HOSTNAME="192.168.x.x"
   
   # On Mac/Linux:
   export REACT_NATIVE_PACKAGER_HOSTNAME="192.168.x.x"
   ```

4. **Start the Expo development server**:
   ```bash
   npx expo start -c
   ```
   - Scan the QR code with Expo Go app
   - The app will connect to your development server

5. **Update API Endpoints**:
   - Open `src/app/(tabs)/detection.tsx`
   - Find and replace `http://192.168.29.32:5001` with `http://YOUR_IP:5001`
   - This ensures the mobile app can connect to your backend server

#### Terminal 2: Start the Backend Prediction API
1. Open a new terminal window.
2. Navigate to the backend directory:
   ```bash
   cd src
   cd send
   ```
3. (First time only) Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend server:
   ```bash
   python predict_api.py
   ```

- The backend must be running for the app to make predictions on audio.
- Make sure both terminals remain open while using the app.

### 5. Running Tests
```bash
npm test
```

### 6. Project Structure
- `src/app/` — Main application code (web & mobile)
- `src/components/` — Reusable UI components
- `src/ML_model/` — Pre-trained ML models and scripts
- `src/backend/` — Backend scripts/services (if any)
- `src/test_audio/` — Test audio files and scripts

### 7. Example Usage
- On launch, the app will start listening for environmental sounds.
- Detected events (e.g., Airport, Bus, Park) will be displayed wuth the Confidence.

### 8. Troubleshooting
- If you face issues with dependencies, try deleting `node_modules` and running `npm install` again.
- For mobile issues, ensure your device and computer are on the same Wi-Fi network and have IP address as that of your computer.

---

