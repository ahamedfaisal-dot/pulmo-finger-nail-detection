#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <Adafruit_MLX90614.h>

// ================= WIFI =================
const char* ssid     = "Don't smuggle_2G";
const char* password = "azam1502";

// ================= SERVER =================
const char* serverIP = "192.168.0.200";   // Raspberry Pi IP
const int serverPort = 5000;

// ================= SENSOR OBJECTS =================
MAX30105 particleSensor;
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// ================= HEART RATE =================
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;

float beatsPerMinute;
int beatAvg = 0;

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\nESP8266 Sensor Node Starting...");

  // I2C Start
  Wire.begin(D2, D1);
  Wire.setClock(100000);

  // MAX30102 Init
  Serial.println("Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found!");
    while (1);
  }

  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);

  Serial.println("MAX30102 Ready");

  // MLX90614 Init
  Serial.println("Initializing MLX90614...");
  if (!mlx.begin()) {
    Serial.println("MLX90614 not found!");
    while (1);
  }

  Serial.println("MLX90614 Ready");

  // WiFi Connect
  connectWiFi();
}

// ================= LOOP =================
void loop() {

  // Reconnect WiFi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  long irValue = particleSensor.getIR();

  // Beat Detection
  if (checkForBeat(irValue)) {
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 255 && beatsPerMinute > 20) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;

      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++) {
        beatAvg += rates[x];
      }
      beatAvg /= RATE_SIZE;
    }
  }

  // Report every 2 sec
  static unsigned long lastReport = 0;
  if (millis() - lastReport > 2000) {
    lastReport = millis();

    float objTemp = mlx.readObjectTempC();

    int hrToSend = 0;
    int spo2ToSend = 0;

    if (irValue > 50000) {   // Finger detected
      hrToSend = beatAvg;
      spo2ToSend = 98;       // placeholder
    }

    Serial.print("IR: ");
    Serial.print(irValue);
    Serial.print(" | Temp: ");
    Serial.print(objTemp);
    Serial.print(" C | HR: ");
    Serial.print(hrToSend);
    Serial.print(" | SpO2: ");
    Serial.println(spo2ToSend);

    sendToPi(objTemp, hrToSend, spo2ToSend);
  }
}

// ================= WIFI FUNCTION =================
void connectWiFi() {
  Serial.print("Connecting to WiFi");

  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi Failed!");
  }
}

// ================= SEND DATA =================
void sendToPi(float t, int h, int s) {
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClient client;
    HTTPClient http;

    String url = "http://" + String(serverIP) + ":" + String(serverPort) + "/update_sensors";

    Serial.println("Sending to: " + url);

    http.begin(client, url);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"temperature\":" + String(t, 1) +
                     ",\"heart_rate\":" + String(h) +
                     ",\"spo2\":" + String(s) + "}";

    int httpCode = http.POST(payload);

    if (httpCode > 0) {
      Serial.printf("HTTP Response code: %d\n", httpCode);
      String response = http.getString();
      Serial.println(response);
    } else {
      Serial.printf("HTTP POST failed: %s\n", http.errorToString(httpCode).c_str());
    }

    http.end();
  }
}