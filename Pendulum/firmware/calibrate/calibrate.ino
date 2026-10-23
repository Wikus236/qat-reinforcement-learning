#include <Wire.h>

constexpr uint8_t ENCODER_SDA_PIN = 21;
constexpr uint8_t ENCODER_SCL_PIN = 22;

constexpr uint8_t AS5600_ADDRESS = 0x36;
constexpr uint8_t STATUS_REGISTER = 0x0B;
constexpr uint8_t RAW_ANGLE_REGISTER = 0x0C;
constexpr uint8_t STATUS_MAGNET_DETECTED = 1 << 5;
constexpr uint8_t STATUS_MAGNET_TOO_WEAK = 1 << 4;
constexpr uint8_t STATUS_MAGNET_TOO_STRONG = 1 << 3;

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t SAMPLE_PERIOD_US = 5000;  // 200 Hz
constexpr uint32_t CALIBRATION_TIME_MS = 2000;
constexpr uint32_t MEASUREMENT_TIMEOUT_MS = 20000;

constexpr float GRAVITY_MPS2 = 9.81f;
constexpr float FILTER_ALPHA = 0.25f;
constexpr float LIFT_THRESHOLD_DEGREES = 15.0f;
constexpr float RELEASE_DROP_DEGREES = 2.0f;
constexpr float RELEASE_VELOCITY_DPS = 10.0f;
constexpr float DIRECTION_VELOCITY_DPS = 4.0f;
constexpr float MIN_PEAK_DEGREES = 1.5f;
constexpr int REQUIRED_PEAK_COUNT = 7;

enum class MeasurementState {
  CALIBRATING,
  WAITING_FOR_LIFT,
  ARMED,
  MEASURING,
  COMPLETE,
};

MeasurementState state = MeasurementState::CALIBRATING;

uint32_t lastSampleUs = 0;
uint32_t calibrationStartMs = 0;
uint32_t measurementStartMs = 0;
double calibrationSinSum = 0.0;
double calibrationCosSum = 0.0;
uint32_t calibrationSamples = 0;

float bottomAngleDegrees = 0.0f;
float filteredAngleDegrees = 0.0f;
float previousFilteredAngleDegrees = 0.0f;
float angularVelocityDps = 0.0f;
float maximumLiftAmplitudeDegrees = 0.0f;

int motionDirection = 0;
float candidatePeakDegrees = 0.0f;
float peakAmplitudesDegrees[REQUIRED_PEAK_COUNT];
uint32_t peakTimesMs[REQUIRED_PEAK_COUNT];
int peakCount = 0;

bool readRegister8(uint8_t registerAddress, uint8_t &value) {
  Wire.beginTransmission(AS5600_ADDRESS);
  Wire.write(registerAddress);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(AS5600_ADDRESS, static_cast<uint8_t>(1)) != 1) {
    return false;
  }

  value = Wire.read();
  return true;
}

bool readRegister12(uint8_t registerAddress, uint16_t &value) {
  Wire.beginTransmission(AS5600_ADDRESS);
  Wire.write(registerAddress);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(AS5600_ADDRESS, static_cast<uint8_t>(2)) != 2) {
    return false;
  }

  const uint8_t highByte = Wire.read();
  const uint8_t lowByte = Wire.read();
  value = (static_cast<uint16_t>(highByte & 0x0F) << 8) | lowByte;
  return true;
}

bool sensorIsConnected() {
  Wire.beginTransmission(AS5600_ADDRESS);
  return Wire.endTransmission() == 0;
}

bool magnetIsValid() {
  uint8_t status = 0;
  if (!readRegister8(STATUS_REGISTER, status)) {
    return false;
  }

  return (status & STATUS_MAGNET_DETECTED) != 0 &&
         (status & STATUS_MAGNET_TOO_WEAK) == 0 &&
         (status & STATUS_MAGNET_TOO_STRONG) == 0;
}

bool readRawAngle(uint16_t &rawAngle) {
  return readRegister12(RAW_ANGLE_REGISTER, rawAngle);
}

float rawAngleToDegrees(uint16_t rawAngle) {
  return static_cast<float>(rawAngle) * 360.0f / 4096.0f;
}

float wrapDegrees(float angleDegrees) {
  while (angleDegrees > 180.0f) {
    angleDegrees -= 360.0f;
  }
  while (angleDegrees <= -180.0f) {
    angleDegrees += 360.0f;
  }
  return angleDegrees;
}

float angleFromBottom(float absoluteAngleDegrees) {
  return wrapDegrees(absoluteAngleDegrees - bottomAngleDegrees);
}

void resetMeasurement() {
  state = MeasurementState::CALIBRATING;
  calibrationStartMs = millis();
  calibrationSinSum = 0.0;
  calibrationCosSum = 0.0;
  calibrationSamples = 0;
  maximumLiftAmplitudeDegrees = 0.0f;
  motionDirection = 0;
  candidatePeakDegrees = 0.0f;
  peakCount = 0;

  Serial.println();
  Serial.println("Pendulum measurement reset.");
  Serial.println("Leave the pendulum hanging still for 2 seconds...");
}

void finishCalibration() {
  const float meanRadians = atan2(
      static_cast<float>(calibrationSinSum),
      static_cast<float>(calibrationCosSum));
  bottomAngleDegrees = meanRadians * 180.0f / PI;
  if (bottomAngleDegrees < 0.0f) {
    bottomAngleDegrees += 360.0f;
  }

  filteredAngleDegrees = 0.0f;
  previousFilteredAngleDegrees = 0.0f;
  angularVelocityDps = 0.0f;
  state = MeasurementState::WAITING_FOR_LIFT;

  Serial.print("Bottom angle calibrated at ");
  Serial.print(bottomAngleDegrees, 2);
  Serial.println(" degrees.");
  Serial.println("Lift the pendulum at least 15 degrees, then release it.");
}

void recordPeak(float peakDegrees, uint32_t peakTimeMs) {
  const float amplitude = fabsf(peakDegrees);
  if (amplitude < MIN_PEAK_DEGREES || peakCount >= REQUIRED_PEAK_COUNT) {
    return;
  }

  peakAmplitudesDegrees[peakCount] = amplitude;
  peakTimesMs[peakCount] = peakTimeMs;
  peakCount++;

  Serial.print("Peak ");
  Serial.print(peakCount);
  Serial.print(": ");
  Serial.print(peakDegrees, 2);
  Serial.print(" deg at ");
  Serial.print((peakTimeMs - measurementStartMs) / 1000.0f, 3);
  Serial.println(" s");
}

void printResults() {
  if (peakCount < 3) {
    Serial.println("Not enough peaks were detected. Repeat with a larger release angle.");
    state = MeasurementState::COMPLETE;
    return;
  }

  const float halfCycleCount = static_cast<float>(peakCount - 1);
  const float elapsedSeconds =
      (peakTimesMs[peakCount - 1] - peakTimesMs[0]) / 1000.0f;
  const float periodSeconds = 2.0f * elapsedSeconds / halfCycleCount;

  float meanPeakRadians = 0.0f;
  for (int index = 0; index < peakCount; index++) {
    meanPeakRadians += peakAmplitudesDegrees[index] * PI / 180.0f;
  }
  meanPeakRadians /= static_cast<float>(peakCount);

  const float angleSquared = meanPeakRadians * meanPeakRadians;
  const float periodCorrection =
      1.0f +
      angleSquared / 16.0f +
      11.0f * angleSquared * angleSquared / 3072.0f +
      173.0f * angleSquared * angleSquared * angleSquared / 737280.0f;
  const float smallAnglePeriod = periodSeconds / periodCorrection;
  const float effectiveLength =
      GRAVITY_MPS2 * powf(smallAnglePeriod / (2.0f * PI), 2.0f);

  // Fit ln(amplitude) against elapsed full cycles. The negative slope is the
  // logarithmic decrement per complete oscillation.
  float sumX = 0.0f;
  float sumY = 0.0f;
  float sumXX = 0.0f;
  float sumXY = 0.0f;
  for (int index = 0; index < peakCount; index++) {
    const float cycles = 0.5f * static_cast<float>(index);
    const float logAmplitude = logf(peakAmplitudesDegrees[index]);
    sumX += cycles;
    sumY += logAmplitude;
    sumXX += cycles * cycles;
    sumXY += cycles * logAmplitude;
  }

  const float count = static_cast<float>(peakCount);
  const float denominator = count * sumXX - sumX * sumX;
  float logarithmicDecrement = 0.0f;
  if (fabsf(denominator) > 1.0e-6f) {
    const float slope = (count * sumXY - sumX * sumY) / denominator;
    logarithmicDecrement = fmaxf(0.0f, -slope);
  }

  const float dampingRatio = logarithmicDecrement /
      sqrtf(4.0f * PI * PI + logarithmicDecrement * logarithmicDecrement);
  const float simulationDamping =
      2.0f * dampingRatio * sqrtf(GRAVITY_MPS2 * effectiveLength);

  Serial.println();
  Serial.println("========== MEASUREMENT RESULTS ==========");
  Serial.print("Detected peaks:             ");
  Serial.println(peakCount);
  Serial.print("Measured period:            ");
  Serial.print(periodSeconds, 4);
  Serial.println(" s");
  Serial.print("Large-angle correction:     ");
  Serial.println(periodCorrection, 4);
  Serial.print("Small-angle period:         ");
  Serial.print(smallAnglePeriod, 4);
  Serial.println(" s");
  Serial.print("Effective pendulum length:  ");
  Serial.print(effectiveLength, 4);
  Serial.println(" m");
  Serial.print("Logarithmic decrement:      ");
  Serial.println(logarithmicDecrement, 5);
  Serial.print("Damping ratio:              ");
  Serial.println(dampingRatio, 5);
  Serial.print("Simulation pendulum_damping:");
  Serial.println(simulationDamping, 5);
  Serial.println("=========================================");
  Serial.println("Press R to measure again.");

  state = MeasurementState::COMPLETE;
}

void updateMeasurement(float angleDegrees, float velocityDps) {
  const float amplitude = fabsf(angleDegrees);

  if (state == MeasurementState::WAITING_FOR_LIFT) {
    if (amplitude >= LIFT_THRESHOLD_DEGREES) {
      maximumLiftAmplitudeDegrees = amplitude;
      state = MeasurementState::ARMED;
      Serial.println("Lift detected. Release the pendulum...");
    }
    return;
  }

  if (state == MeasurementState::ARMED) {
    maximumLiftAmplitudeDegrees = fmaxf(maximumLiftAmplitudeDegrees, amplitude);
    const bool movingTowardBottom = angleDegrees * velocityDps < 0.0f;
    const bool movedFromMaximum =
        amplitude <= maximumLiftAmplitudeDegrees - RELEASE_DROP_DEGREES;

    if (movingTowardBottom && movedFromMaximum &&
        fabsf(velocityDps) >= RELEASE_VELOCITY_DPS) {
      measurementStartMs = millis();
      motionDirection = velocityDps > 0.0f ? 1 : -1;
      candidatePeakDegrees = angleDegrees;
      peakCount = 0;
      state = MeasurementState::MEASURING;
      Serial.println("Release detected. Measuring free oscillation...");
    }
    return;
  }

  if (state != MeasurementState::MEASURING) {
    return;
  }

  if (motionDirection > 0) {
    candidatePeakDegrees = fmaxf(candidatePeakDegrees, angleDegrees);
    if (velocityDps < -DIRECTION_VELOCITY_DPS) {
      recordPeak(candidatePeakDegrees, millis());
      motionDirection = -1;
      candidatePeakDegrees = angleDegrees;
    }
  } else {
    candidatePeakDegrees = fminf(candidatePeakDegrees, angleDegrees);
    if (velocityDps > DIRECTION_VELOCITY_DPS) {
      recordPeak(candidatePeakDegrees, millis());
      motionDirection = 1;
      candidatePeakDegrees = angleDegrees;
    }
  }

  if (peakCount >= REQUIRED_PEAK_COUNT) {
    printResults();
  } else if (millis() - measurementStartMs >= MEASUREMENT_TIMEOUT_MS) {
    Serial.println("Measurement timed out.");
    printResults();
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin(ENCODER_SDA_PIN, ENCODER_SCL_PIN);
  Wire.setClock(400000);
  delay(250);

  Serial.println("AS5600 pendulum period and damping measurement");
  Serial.println("I2C pins: SDA = 21, SCL = 22");

  if (!sensorIsConnected()) {
    Serial.println("ERROR: AS5600 not detected at address 0x36.");
    while (true) {
      delay(1000);
    }
  }

  if (!magnetIsValid()) {
    Serial.println("ERROR: AS5600 magnet is missing, too weak, or too strong.");
    while (true) {
      delay(1000);
    }
  }

  resetMeasurement();
}

void loop() {
  if (Serial.available() > 0) {
    const char command = Serial.read();
    if (command == 'r' || command == 'R') {
      resetMeasurement();
    }
  }

  const uint32_t nowUs = micros();
  if (nowUs - lastSampleUs < SAMPLE_PERIOD_US) {
    return;
  }
  const float dtSeconds =
      lastSampleUs == 0 ? SAMPLE_PERIOD_US / 1000000.0f
                        : (nowUs - lastSampleUs) / 1000000.0f;
  lastSampleUs = nowUs;

  uint16_t rawAngle = 0;
  if (!readRawAngle(rawAngle)) {
    Serial.println("ERROR: Could not read AS5600 angle.");
    return;
  }

  const float absoluteAngleDegrees = rawAngleToDegrees(rawAngle);

  if (state == MeasurementState::CALIBRATING) {
    const float radians = absoluteAngleDegrees * PI / 180.0f;
    calibrationSinSum += sinf(radians);
    calibrationCosSum += cosf(radians);
    calibrationSamples++;

    if (millis() - calibrationStartMs >= CALIBRATION_TIME_MS &&
        calibrationSamples > 0) {
      finishCalibration();
    }
    return;
  }

  const float measuredAngleDegrees = angleFromBottom(absoluteAngleDegrees);
  const float filterDifference =
      wrapDegrees(measuredAngleDegrees - filteredAngleDegrees);
  filteredAngleDegrees =
      wrapDegrees(filteredAngleDegrees + FILTER_ALPHA * filterDifference);
  angularVelocityDps =
      wrapDegrees(filteredAngleDegrees - previousFilteredAngleDegrees) /
      dtSeconds;
  previousFilteredAngleDegrees = filteredAngleDegrees;

  updateMeasurement(filteredAngleDegrees, angularVelocityDps);
}
