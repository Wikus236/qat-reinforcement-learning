#include <Wire.h>
#include <soc/gpio_struct.h>
#include "model_data.h"

// Linear pendulum motor controller using an MKS TMC2160-OC driver and AS5600.

// Adjustable upright calibration, in sensor degrees.
// Set to measured top - measured bottom - 180 degrees (wrapped to +/-180).
// The three measured offsets averaged -1.18 degrees; 0 restores the old setting.
constexpr float UPRIGHT_OFFSET_DEGREES = 1.0f;

// Stepper driver pins
constexpr uint8_t ENABLE_PIN = 25;
constexpr uint8_t DIRECTION_PIN = 26;
constexpr uint8_t STEP_PIN = 27;

// Normally-closed end stops. Each switch connects its pin to GND while
// unpressed and opens when pressed. A broken wire is also detected as a limit.
constexpr uint8_t CLOSE_LIMIT_PIN = 32;
constexpr uint8_t FAR_LIMIT_PIN = 33;

// Reserved for the I2C pendulum encoder.
constexpr uint8_t ENCODER_SCL_PIN = 22;
constexpr uint8_t ENCODER_SDA_PIN = 21;

constexpr uint8_t AS5600_ADDRESS = 0x36;
constexpr uint8_t AS5600_STATUS_REGISTER = 0x0B;
constexpr uint8_t AS5600_RAW_ANGLE_REGISTER = 0x0C;
constexpr uint8_t AS5600_MAGNET_DETECTED = 1 << 5;
constexpr uint8_t AS5600_MAGNET_TOO_WEAK = 1 << 4;
constexpr uint8_t AS5600_MAGNET_TOO_STRONG = 1 << 3;
constexpr uint32_t ENCODER_CALIBRATION_SAMPLES = 100;
constexpr uint32_t ENCODER_CALIBRATION_SAMPLE_DELAY_MS = 5;
constexpr float ANGULAR_VELOCITY_FILTER_ALPHA = 0.2f;
constexpr float PI_F = 3.14159265358979323846f;
constexpr float TWO_PI_F = 2.0f * PI_F;

constexpr uint8_t DRIVER_ENABLED = LOW;
constexpr uint8_t DRIVER_DISABLED = HIGH;
constexpr uint8_t LIMIT_TRIGGERED = HIGH;

// Change this to HIGH if calibration moves toward the far switch first.
constexpr uint8_t CLOSE_DIRECTION = LOW;
constexpr uint8_t FAR_DIRECTION = CLOSE_DIRECTION == LOW ? HIGH : LOW;

constexpr uint32_t STEP_HALF_PERIOD_US = 500;
constexpr uint32_t START_DELAY_MS = 3000;
constexpr uint32_t DIRECTION_PAUSE_MS = 500;
constexpr uint32_t LIMIT_RECOVERY_PAUSE_MS = 250;
constexpr uint32_t LIMIT_DEBOUNCE_MS = 20;
constexpr uint32_t TRANSIENT_LIMIT_RECHECK_MS = 100;
constexpr uint32_t LIMIT_POSITION_MARGIN_DIVISOR = 10;
constexpr uint32_t CLEARANCE_STEPS = 100;
constexpr uint32_t MAX_CLEAR_STEPS = 1000;
constexpr uint32_t MAX_TRAVEL_STEPS = 100000;

// The trained environment uses one control update every 10 ms. Calibration
// keeps the proven 1000-pulse/s rate; a timer maintains policy STEP pulses.
constexpr uint32_t CONTROL_PERIOD_MS = 10;
constexpr int32_t MAX_POLICY_STEP_RATE_PPS = 10000;
constexpr int32_t POLICY_ACCELERATION_PPS2 = 80000;
constexpr int32_t POLICY_RATE_CHANGE_PER_PERIOD =
    POLICY_ACCELERATION_PPS2 * CONTROL_PERIOD_MS / 1000;
constexpr uint32_t STEP_TIMER_TICK_US = 50;
constexpr uint32_t STEP_TIMER_TICKS_PER_SECOND = 1000000 / STEP_TIMER_TICK_US;
// Pulses pause immediately; only a sustained signal resets the controller.
constexpr uint32_t TIMER_LIMIT_CONFIRM_TICKS =
    (LIMIT_DEBOUNCE_MS * 1000 + STEP_TIMER_TICK_US - 1) / STEP_TIMER_TICK_US;
constexpr uint32_t STATUS_PERIOD_MS = 1000;
// Long Serial status lines delay DQN decisions even though STEP keeps running.
// Enable this only for short diagnostic runs.
constexpr bool ENABLE_RUNTIME_STATUS = false;
constexpr uint32_t DQN_DECISION_PERIOD_MS = 10;
constexpr float RAIL_LENGTH_METRES = 0.8f;
// Match the encoder orientation verified by sketch_sep21a's balance control.
// Positive angle means the pendulum leans toward the FAR end of the rail.
constexpr float MODEL_PENDULUM_SIGN = 1.0f;
constexpr int MODEL_MAX_NEURONS = 128;

static_assert(MODEL_INPUT_SIZE == 4, "Pendulum DQN must have four inputs");
static_assert(MODEL_OUTPUT_SIZE == 3, "Pendulum DQN must have three actions");

uint32_t railLengthSteps = 0;
volatile int32_t currentPositionSteps = 0;
bool controllerReady = false;
uint32_t lastStatusTime = 0;
uint8_t currentAction = 1;
int32_t currentPolicyStepRatePps = 0;
hw_timer_t *stepTimer = nullptr;
portMUX_TYPE stepTimerMux = portMUX_INITIALIZER_UNLOCKED;
volatile int32_t timerStepRatePps = 0;
volatile uint32_t timerStepPhase = 0;
volatile bool timerStepHigh = false;
volatile bool timerLimitHalt = false;
volatile uint32_t timerCloseLimitTicks = 0;
volatile uint32_t timerFarLimitTicks = 0;
uint8_t timerDirection = CLOSE_DIRECTION;
uint32_t lastControlTimeMs = 0;
float encoderBottomReferenceRadians = 0.0f;
float pendulumAngleRadians = -PI_F;
float pendulumAngularVelocityRadPerSec = 0.0f;
float previousPendulumAngleRadians = -PI_F;
uint32_t previousEncoderSampleTimeUs = 0;
int8_t railSafetyOverrideDirection = 0;
float modelValuesA[MODEL_MAX_NEURONS];
float modelValuesB[MODEL_MAX_NEURONS];
float modelState[MODEL_INPUT_SIZE] = {};
float modelQValues[MODEL_OUTPUT_SIZE] = {};
uint32_t modelInferenceTimeUs = 0;
uint32_t lastDqnDecisionTimeMs = 0;
uint32_t dqnActionSwitchCount = 0;
bool dqnHasDecision = false;


// Buffer samples during control; print only after stopping the motor.
constexpr uint32_t LOG_DURATION_US = 30000000;
constexpr uint16_t LOG_MAX_SAMPLES = 3002;
constexpr float LOG_PENDULUM_LENGTH_M = 0.18f;
struct PositionSample {
  uint32_t elapsedUs;
  int32_t positionSteps;
  int32_t stepRatePps;
  float angleRad;
  float angularVelocityRadS;
  uint8_t action;
  uint8_t atRailLimit;
  int8_t safetyOverride;
};
PositionSample positionSamples[LOG_MAX_SAMPLES];
uint16_t positionSampleCount = 0;
uint32_t positionLogStartUs = 0;
bool positionLogStarted = false;
bool positionLogSent = false;

void recordPositionSample(uint32_t sampleUs) {
  uint32_t elapsed = sampleUs - positionLogStartUs;
  if (elapsed > LOG_DURATION_US || positionSampleCount >= LOG_MAX_SAMPLES) {
    return;
  }
  PositionSample &sample = positionSamples[positionSampleCount++];
  sample.elapsedUs = elapsed;
  portENTER_CRITICAL(&stepTimerMux);
  sample.positionSteps = currentPositionSteps;
  portEXIT_CRITICAL(&stepTimerMux);
  sample.stepRatePps = currentPolicyStepRatePps;
  sample.angleRad = wrapRadians(MODEL_PENDULUM_SIGN * pendulumAngleRadians);
  sample.angularVelocityRadS = MODEL_PENDULUM_SIGN * pendulumAngularVelocityRadPerSec;
  sample.action = currentAction;
  sample.atRailLimit = timerLimitHalt ||
      digitalRead(CLOSE_LIMIT_PIN) == LIMIT_TRIGGERED ||
      digitalRead(FAR_LIMIT_PIN) == LIMIT_TRIGGERED ||
      sample.positionSteps <= 0 ||
      sample.positionSteps >= static_cast<int32_t>(railLengthSteps);
  sample.safetyOverride = railSafetyOverrideDirection;
}

void sendPositionLog() {
  if (!positionLogStarted || positionLogSent) {
    return;
  }
  positionLogSent = true;
  Serial.println("CSV_BEGIN");
  Serial.println("time_s,cart_position_m,cart_velocity_m_s,angle_rad,angle_deg,angular_velocity_rad_s,tip_x_m,tip_y_m,at_rail_limit,action_to_next_step,motor_command_to_next_step,safety_override");
  const float metresPerStep = RAIL_LENGTH_METRES / railLengthSteps;
  for (uint16_t i = 0; i < positionSampleCount; ++i) {
    const PositionSample &sample = positionSamples[i];
    float x = (sample.positionSteps - 0.5f * railLengthSteps) * metresPerStep;
    float velocity = sample.stepRatePps * metresPerStep;
    Serial.printf("%.6f,%.6f,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%u,%u,%d,%d\n",
        sample.elapsedUs * 1e-6, x, velocity, sample.angleRad,
        sample.angleRad * 180.0f / PI_F, sample.angularVelocityRadS,
        x + LOG_PENDULUM_LENGTH_M * sinf(sample.angleRad),
        LOG_PENDULUM_LENGTH_M * cosf(sample.angleRad),
        static_cast<unsigned>(sample.atRailLimit),
        static_cast<unsigned>(sample.action), static_cast<int>(sample.action) - 1,
        static_cast<int>(sample.safetyOverride));
  }
  Serial.println("CSV_END");
  Serial.printf("Recorded %u samples. Cart position/velocity are step-based estimates; tip coordinates are derived.\n", positionSampleCount);
}

bool readAs5600Register8(uint8_t registerAddress, uint8_t &value) {
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

bool readAs5600RawAngle(uint16_t &rawAngle) {
  Wire.beginTransmission(AS5600_ADDRESS);
  Wire.write(AS5600_RAW_ANGLE_REGISTER);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(AS5600_ADDRESS, static_cast<uint8_t>(2)) != 2) {
    return false;
  }

  uint8_t highByte = Wire.read();
  uint8_t lowByte = Wire.read();
  rawAngle = (static_cast<uint16_t>(highByte & 0x0F) << 8) | lowByte;
  return true;
}

float wrapRadians(float angle) {
  while (angle >= PI_F) {
    angle -= TWO_PI_F;
  }
  while (angle < -PI_F) {
    angle += TWO_PI_F;
  }
  return angle;
}

const char *as5600MagnetStatusText(uint8_t status) {
  if ((status & AS5600_MAGNET_DETECTED) == 0) {
    return "NOT DETECTED";
  }
  if ((status & AS5600_MAGNET_TOO_WEAK) != 0) {
    return "TOO WEAK";
  }
  if ((status & AS5600_MAGNET_TOO_STRONG) != 0) {
    return "TOO STRONG";
  }
  return "OK";
}

bool initializePendulumEncoder() {
  Wire.begin(ENCODER_SDA_PIN, ENCODER_SCL_PIN);
  Wire.setClock(400000);
  delay(100);

  Wire.beginTransmission(AS5600_ADDRESS);
  if (Wire.endTransmission() != 0) {
    Serial.println("AS5600 initialization failed: sensor not detected.");
    return false;
  }

  uint8_t status = 0;
  if (!readAs5600Register8(AS5600_STATUS_REGISTER, status)) {
    Serial.println("AS5600 initialization failed: status read failed.");
    return false;
  }

  Serial.print("AS5600 magnet status: ");
  Serial.println(as5600MagnetStatusText(status));

  bool magnetIsValid =
      (status & AS5600_MAGNET_DETECTED) != 0 &&
      (status & AS5600_MAGNET_TOO_WEAK) == 0 &&
      (status & AS5600_MAGNET_TOO_STRONG) == 0;
  if (!magnetIsValid) {
    return false;
  }

  // Use a circular mean so calibration also works near the 0/360-degree wrap.
  float sineSum = 0.0f;
  float cosineSum = 0.0f;
  for (uint32_t sample = 0; sample < ENCODER_CALIBRATION_SAMPLES; ++sample) {
    uint16_t rawAngle = 0;
    if (!readAs5600RawAngle(rawAngle)) {
      Serial.println("AS5600 initialization failed: angle read failed.");
      return false;
    }

    float sensorAngle =
        static_cast<float>(rawAngle) * TWO_PI_F / 4096.0f;
    sineSum += sinf(sensorAngle);
    cosineSum += cosf(sensorAngle);
    delay(ENCODER_CALIBRATION_SAMPLE_DELAY_MS);
  }

  encoderBottomReferenceRadians = atan2f(sineSum, cosineSum);
  if (encoderBottomReferenceRadians < 0.0f) {
    encoderBottomReferenceRadians += TWO_PI_F;
  }

  pendulumAngleRadians = -PI_F;
  previousPendulumAngleRadians = pendulumAngleRadians;
  pendulumAngularVelocityRadPerSec = 0.0f;
  previousEncoderSampleTimeUs = 0;

  Serial.print("Pendulum bottom reference: ");
  Serial.print(encoderBottomReferenceRadians * 180.0f / PI_F, 2);
  Serial.println(" degrees");
  Serial.print("Upright offset: ");
  Serial.print(UPRIGHT_OFFSET_DEGREES, 2);
  Serial.println(" degrees");
  return true;
}

bool updatePendulumState() {
  uint16_t rawAngle = 0;
  if (!readAs5600RawAngle(rawAngle)) {
    return false;
  }

  uint32_t nowUs = micros();
  float sensorAngle = static_cast<float>(rawAngle) * TWO_PI_F / 4096.0f;
  float angleFromBottom =
      wrapRadians(sensorAngle - encoderBottomReferenceRadians);
  float newPendulumAngle = wrapRadians(
      angleFromBottom - PI_F - UPRIGHT_OFFSET_DEGREES * PI_F / 180.0f);

  if (previousEncoderSampleTimeUs != 0) {
    float dt = static_cast<float>(nowUs - previousEncoderSampleTimeUs) * 1e-6f;
    if (dt > 0.0f) {
      float angleChange =
          wrapRadians(newPendulumAngle - previousPendulumAngleRadians);
      float instantaneousVelocity = angleChange / dt;
      pendulumAngularVelocityRadPerSec =
          ANGULAR_VELOCITY_FILTER_ALPHA * instantaneousVelocity +
          (1.0f - ANGULAR_VELOCITY_FILTER_ALPHA) *
              pendulumAngularVelocityRadPerSec;
    }
  } else {
    pendulumAngularVelocityRadPerSec = 0.0f;
  }

  pendulumAngleRadians = newPendulumAngle;
  previousPendulumAngleRadians = newPendulumAngle;
  previousEncoderSampleTimeUs = nowUs;
  return true;
}

bool limitIsTriggered(uint8_t pin) {
  if (digitalRead(pin) != LIMIT_TRIGGERED) {
    return false;
  }

  // No pulse is generated while a continuously active signal is confirmed.
  uint32_t confirmationStart = millis();
  while (millis() - confirmationStart < LIMIT_DEBOUNCE_MS) {
    if (digitalRead(pin) != LIMIT_TRIGGERED) {
      return false;
    }
    delay(1);
  }

  return true;
}

bool anyLimitIsTriggered() {
  return limitIsTriggered(CLOSE_LIMIT_PIN) ||
         limitIsTriggered(FAR_LIMIT_PIN);
}

void setDirection(uint8_t direction) {
  digitalWrite(DIRECTION_PIN, direction);
  timerDirection = direction;
  delayMicroseconds(10);
}

void stepOnce() {
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(STEP_HALF_PERIOD_US);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(STEP_HALF_PERIOD_US);
}

void IRAM_ATTR onStepTimer() {
  portENTER_CRITICAL_ISR(&stepTimerMux);
  if (timerStepHigh) {
    GPIO.out_w1tc = 1UL << STEP_PIN;
    timerStepHigh = false;
  }

  int32_t rate = timerStepRatePps;
  if (rate != 0) {
    const uint32_t limitInputs = GPIO.in1.data;
    const bool closeActive = (limitInputs & (1UL << (CLOSE_LIMIT_PIN - 32))) != 0;
    const bool farActive = (limitInputs & (1UL << (FAR_LIMIT_PIN - 32))) != 0;
    timerCloseLimitTicks = closeActive ? timerCloseLimitTicks + 1 : 0;
    timerFarLimitTicks = farActive ? timerFarLimitTicks + 1 : 0;

    if (closeActive || farActive) {
      // Do not generate or accumulate pulses while either input is active.
      // A brief glitch keeps the rate and action intact, so they resume as
      // soon as both inputs clear. Confirm each switch independently.
      if (timerCloseLimitTicks >= TIMER_LIMIT_CONFIRM_TICKS ||
          timerFarLimitTicks >= TIMER_LIMIT_CONFIRM_TICKS) {
        timerStepRatePps = 0;
        timerStepPhase = 0;
        timerLimitHalt = true;
      }
    } else {
      uint32_t magnitude = static_cast<uint32_t>(rate < 0 ? -rate : rate);
      timerStepPhase += magnitude;
      if (!timerStepHigh && timerStepPhase >= STEP_TIMER_TICKS_PER_SECOND) {
        int32_t nextPosition = currentPositionSteps + (rate < 0 ? -1 : 1);
        if (nextPosition < 0 ||
            nextPosition > static_cast<int32_t>(railLengthSteps)) {
          timerStepRatePps = 0;
          timerStepPhase = 0;
          timerLimitHalt = true;
        } else {
          timerStepPhase -= STEP_TIMER_TICKS_PER_SECOND;
          GPIO.out_w1ts = 1UL << STEP_PIN;
          timerStepHigh = true;
          currentPositionSteps = nextPosition;
        }
      }
    }
  } else {
    timerCloseLimitTicks = 0;
    timerFarLimitTicks = 0;
  }
  portEXIT_CRITICAL_ISR(&stepTimerMux);
}

void setTimerStepRate(int32_t rate) {
  portENTER_CRITICAL(&stepTimerMux);
  timerStepRatePps = timerLimitHalt ? 0 : rate;
  if (timerStepRatePps == 0) {
    timerStepPhase = 0;
  }
  portEXIT_CRITICAL(&stepTimerMux);
}

void stopMotor() {
  portENTER_CRITICAL(&stepTimerMux);
  timerStepRatePps = 0;
  timerStepPhase = 0;
  timerStepHigh = false;
  timerCloseLimitTicks = 0;
  timerFarLimitTicks = 0;
  GPIO.out_w1tc = 1UL << STEP_PIN;
  portEXIT_CRITICAL(&stepTimerMux);
}

void disableMotor() {
  stopMotor();
  digitalWrite(ENABLE_PIN, DRIVER_DISABLED);
}

void stopWithError(const char *message) {
  controllerReady = false;
  disableMotor();
  sendPositionLog();
  Serial.print("Controller stopped: ");
  Serial.println(message);
  Serial.print("Diagnostic | position: ");
  Serial.print(currentPositionSteps);
  Serial.print(" / ");
  Serial.print(railLengthSteps);
  Serial.print(" | close raw: ");
  Serial.print(digitalRead(CLOSE_LIMIT_PIN));
  Serial.print(" | far raw: ");
  Serial.println(digitalRead(FAR_LIMIT_PIN));
}

bool moveStepsSafely(uint8_t direction, uint32_t numberOfSteps) {
  setDirection(direction);

  for (uint32_t step = 0; step < numberOfSteps; ++step) {
    if (anyLimitIsTriggered()) {
      stopMotor();
      return false;
    }

    stepOnce();

    if (anyLimitIsTriggered()) {
      stopMotor();
      return false;
    }
  }

  return true;
}

bool moveOffLimit(uint8_t activeLimitPin, uint8_t oppositeLimitPin,
                  uint8_t direction) {
  setDirection(direction);

  uint32_t steps = 0;
  while (limitIsTriggered(activeLimitPin) && steps < MAX_CLEAR_STEPS) {
    if (limitIsTriggered(oppositeLimitPin)) {
      return false;
    }
    stepOnce();
    ++steps;
  }

  if (limitIsTriggered(activeLimitPin)) {
    return false;
  }

  return moveStepsSafely(direction, CLEARANCE_STEPS);
}

bool prepareUntriggeredPosition() {
  bool closeTriggered = limitIsTriggered(CLOSE_LIMIT_PIN);
  bool farTriggered = limitIsTriggered(FAR_LIMIT_PIN);

  if (closeTriggered && farTriggered) {
    return false;
  }

  if (closeTriggered) {
    Serial.println("Moving away from the close switch...");
    return moveOffLimit(CLOSE_LIMIT_PIN, FAR_LIMIT_PIN, FAR_DIRECTION);
  }

  if (farTriggered) {
    Serial.println("Moving away from the far switch...");
    return moveOffLimit(FAR_LIMIT_PIN, CLOSE_LIMIT_PIN, CLOSE_DIRECTION);
  }

  return true;
}

bool moveUntilLimit(uint8_t targetLimitPin, uint8_t oppositeLimitPin,
                    uint8_t direction, uint32_t maximumSteps,
                    uint32_t &stepsMoved) {
  setDirection(direction);
  stepsMoved = 0;
  bool allowInitialOppositeLimit = limitIsTriggered(oppositeLimitPin);

  while (stepsMoved < maximumSteps) {
    if (limitIsTriggered(targetLimitPin)) {
      stopMotor();
      return true;
    }

    if (limitIsTriggered(oppositeLimitPin) &&
        (!allowInitialOppositeLimit || stepsMoved >= MAX_CLEAR_STEPS)) {
      stopMotor();
      return false;
    }

    stepOnce();
    ++stepsMoved;
  }

  stopMotor();
  return false;
}

// Move a known distance away from an active endpoint. The starting switch may
// remain active briefly, but must release within MAX_CLEAR_STEPS.
bool moveFromEndpoint(uint8_t startingLimitPin, uint8_t oppositeLimitPin,
                      uint8_t direction, uint32_t numberOfSteps) {
  setDirection(direction);
  bool startingLimitReleased = !limitIsTriggered(startingLimitPin);

  for (uint32_t step = 0; step < numberOfSteps; ++step) {
    if (limitIsTriggered(oppositeLimitPin)) {
      stopMotor();
      return false;
    }

    if (!limitIsTriggered(startingLimitPin)) {
      startingLimitReleased = true;
    } else if (step >= MAX_CLEAR_STEPS) {
      stopMotor();
      return false;
    }

    stepOnce();
  }

  stopMotor();
  return startingLimitReleased && !anyLimitIsTriggered();
}

bool moveFromFarToMiddle() {
  uint32_t stepsFromFarToMiddle = railLengthSteps - railLengthSteps / 2;

  Serial.print("Moving ");
  Serial.print(stepsFromFarToMiddle);
  Serial.println(" steps from the far switch to the middle...");

  if (!moveFromEndpoint(FAR_LIMIT_PIN, CLOSE_LIMIT_PIN, CLOSE_DIRECTION,
                        stepsFromFarToMiddle)) {
    return false;
  }

  currentPositionSteps = static_cast<int32_t>(railLengthSteps / 2);
  return true;
}

bool calibrateRail() {
  Serial.println("Rail calibration starting in 3 seconds.");
  delay(START_DELAY_MS);

  digitalWrite(ENABLE_PIN, DRIVER_ENABLED);

  if (!prepareUntriggeredPosition() || anyLimitIsTriggered()) {
    stopWithError("could not start with both limit switches released.");
    return false;
  }

  delay(DIRECTION_PAUSE_MS);
  Serial.println("Finding the close switch...");

  uint32_t stepsToClose = 0;
  if (!moveUntilLimit(CLOSE_LIMIT_PIN, FAR_LIMIT_PIN, CLOSE_DIRECTION,
                      MAX_TRAVEL_STEPS, stepsToClose)) {
    stopWithError("close switch was not found or the far switch was reached.");
    return false;
  }

  currentPositionSteps = 0;
  Serial.print("Close switch found after ");
  Serial.print(stepsToClose);
  Serial.println(" steps. Position set to zero.");

  delay(DIRECTION_PAUSE_MS);
  Serial.println("Finding the far switch and measuring the rail...");

  if (!moveUntilLimit(FAR_LIMIT_PIN, CLOSE_LIMIT_PIN, FAR_DIRECTION,
                      MAX_TRAVEL_STEPS, railLengthSteps)) {
    stopWithError("far switch was not found or the close switch was reached.");
    return false;
  }

  currentPositionSteps = static_cast<int32_t>(railLengthSteps);
  Serial.println("Calibration complete.");
  Serial.print("Rail travel: ");
  Serial.print(railLengthSteps);
  Serial.println(" step pulses");

  delay(DIRECTION_PAUSE_MS);
  if (!moveFromFarToMiddle()) {
    stopWithError("could not move from the far switch to the middle.");
    return false;
  }

  Serial.print("Carriage centred at step ");
  Serial.println(currentPositionSteps);
  return true;
}

bool recoverToMiddle() {
  stopMotor();
  int32_t rateAtLimit = currentPolicyStepRatePps;
  currentPolicyStepRatePps = 0;
  currentAction = 1;
  lastDqnDecisionTimeMs = 0;
  dqnHasDecision = false;
  railSafetyOverrideDirection = 0;
  Serial.println("Limit reached. Policy stopped; returning to the middle.");

  bool closeTriggered = limitIsTriggered(CLOSE_LIMIT_PIN);
  bool farTriggered = limitIsTriggered(FAR_LIMIT_PIN);

  Serial.print("Limit diagnostic | position: ");
  Serial.print(currentPositionSteps);
  Serial.print(" / ");
  Serial.print(railLengthSteps);
  Serial.print(" | rate before stop: ");
  Serial.print(rateAtLimit);
  Serial.print(" | close: ");
  Serial.print(closeTriggered ? "TRIGGERED" : "OK");
  Serial.print(" | far: ");
  Serial.println(farTriggered ? "TRIGGERED" : "OK");

  if (!closeTriggered && !farTriggered) {
    if (currentPositionSteps <= 0 ||
        currentPositionSteps >= static_cast<int32_t>(railLengthSteps)) {
      stopWithError("software rail endpoint reached without a limit switch.");
      return false;
    }
    timerLimitHalt = false;
    Serial.println(
        "Transient limit signal disappeared. Motion remains stopped; "
        "selecting a new action.");
    return true;
  }

  if (closeTriggered && farTriggered) {
    stopWithError("limit recovery could not identify a single endpoint.");
    return false;
  }

  uint32_t positionMargin = railLengthSteps / LIMIT_POSITION_MARGIN_DIVISOR;
  bool positionMatchesClose =
      currentPositionSteps <= static_cast<int32_t>(positionMargin);
  bool positionMatchesFar =
      currentPositionSteps >=
      static_cast<int32_t>(railLengthSteps - positionMargin);
  bool positionMatchesLimit =
      (closeTriggered && positionMatchesClose) ||
      (farTriggered && positionMatchesFar);

  if (!positionMatchesLimit) {
    Serial.println(
        "Limit signal conflicts with the estimated middle position. "
        "Rechecking for electrical noise...");
    delay(TRANSIENT_LIMIT_RECHECK_MS);

    closeTriggered = limitIsTriggered(CLOSE_LIMIT_PIN);
    farTriggered = limitIsTriggered(FAR_LIMIT_PIN);

    if (!closeTriggered && !farTriggered) {
      timerLimitHalt = false;
      Serial.println(
          "False limit trigger rejected. Motion remains stopped; "
          "selecting a new action.");
      return true;
    }

    if (closeTriggered == farTriggered) {
      stopWithError("limit signals remained invalid after recheck.");
      return false;
    }

    stopWithError(
        "limit signal conflicts with the estimated position; check switch "
        "wiring and electrical noise.");
    return false;
  }

  delay(LIMIT_RECOVERY_PAUSE_MS);

  bool recovered = false;
  if (closeTriggered) {
    recovered = moveFromEndpoint(
        CLOSE_LIMIT_PIN,
        FAR_LIMIT_PIN,
        FAR_DIRECTION,
        railLengthSteps / 2);
  } else {
    recovered = moveFromEndpoint(
        FAR_LIMIT_PIN,
        CLOSE_LIMIT_PIN,
        CLOSE_DIRECTION,
        railLengthSteps - railLengthSteps / 2);
  }

  if (!recovered) {
    stopWithError("failed to return to the middle after reaching a limit.");
    return false;
  }

  currentPositionSteps = static_cast<int32_t>(railLengthSteps / 2);
  timerLimitHalt = false;
  Serial.println("Carriage returned to the middle. Full DQN control resumed.");
  return true;
}

int32_t moveRateToward(int32_t currentRate, int32_t targetRate) {
  if (currentRate < targetRate) {
    int32_t nextRate = currentRate + POLICY_RATE_CHANGE_PER_PERIOD;
    return nextRate > targetRate ? targetRate : nextRate;
  }

  if (currentRate > targetRate) {
    int32_t nextRate = currentRate - POLICY_RATE_CHANGE_PER_PERIOD;
    return nextRate < targetRate ? targetRate : nextRate;
  }

  return currentRate;
}

const char *actionName(uint8_t action) {
  if (action == 0) {
    return "CLOSE";
  }
  if (action == 2) {
    return "FAR";
  }
  return "STOP";
}

int32_t targetRateForAction(uint8_t action) {
  if (action == 0) {
    return -MAX_POLICY_STEP_RATE_PPS;
  }
  if (action == 2) {
    return MAX_POLICY_STEP_RATE_PPS;
  }
  return 0;
}

bool runPolicyAction(uint8_t action) {
  if (timerLimitHalt || anyLimitIsTriggered()) {
    return recoverToMiddle();
  }

  int32_t targetRate = targetRateForAction(action);
  int32_t nextRate = moveRateToward(currentPolicyStepRatePps, targetRate);
  // Finish the current direction at zero before changing DIR.
  if ((currentPolicyStepRatePps > 0 && nextRate < 0) ||
      (currentPolicyStepRatePps < 0 && nextRate > 0)) {
    nextRate = 0;
  }

  if (nextRate != 0) {
    uint8_t nextDirection = nextRate < 0 ? CLOSE_DIRECTION : FAR_DIRECTION;
    if (nextDirection != timerDirection) {
      setTimerStepRate(0);
      delayMicroseconds(STEP_TIMER_TICK_US);
      setDirection(nextDirection);
    }
  }

  currentPolicyStepRatePps = nextRate;
  setTimerStepRate(nextRate);
  return true;
}

uint32_t policyBrakingDistanceSteps() {
  int64_t rate = currentPolicyStepRatePps;
  if (rate < 0) {
    rate = -rate;
  }

  return static_cast<uint32_t>(
      rate * rate / (2LL * POLICY_ACCELERATION_PPS2));
}

void printActionChange(const char *reason) {
  Serial.print("[");
  Serial.print(millis());
  Serial.print(" ms] Controller action: ");
  Serial.print(currentAction);
  Serial.print(" (");
  Serial.print(actionName(currentAction));
  Serial.print(") | current rate: ");
  Serial.print(currentPolicyStepRatePps);
  Serial.print(" | target rate: ");
  Serial.print(targetRateForAction(currentAction));
  Serial.print(" | braking distance: ");
  Serial.print(policyBrakingDistanceSteps());
  Serial.print(" steps | reason: ");
  Serial.println(reason);
}

void setControllerAction(uint8_t nextAction, const char *reason) {
  if (nextAction == currentAction) {
    return;
  }

  currentAction = nextAction;
  if (ENABLE_RUNTIME_STATUS) {
    printActionChange(reason);
  }
}

void runModelLayer(const float *input, float *output, int layerIndex) {
  int inputSize = LAYER_INPUT_SIZES[layerIndex];
  int outputSize = LAYER_OUTPUT_SIZES[layerIndex];
  const model_weight_t *weights = LAYER_WEIGHTS[layerIndex];
  const float *bias = LAYER_BIASES[layerIndex];
  float weightScale = LAYER_SCALES[layerIndex];

  for (int outputIndex = 0; outputIndex < outputSize; ++outputIndex) {
    float value = bias[outputIndex];
    int weightOffset = outputIndex * inputSize;

    for (int inputIndex = 0; inputIndex < inputSize; ++inputIndex) {
      float weight = static_cast<float>(weights[weightOffset + inputIndex]);
      if (MODEL_WEIGHTS_ARE_QUANTIZED) {
        weight *= weightScale;
      }
      value += input[inputIndex] * weight;
    }

    output[outputIndex] = value;
  }
}

void applyModelActivation(float *values, int size, int activationIndex) {
  (void)activationIndex;
  // Weight-only QAT: hidden activations remain float after ReLU.
  for (int index = 0; index < size; ++index) {
    values[index] = values[index] > 0.0f ? values[index] : 0.0f;
  }
}

uint8_t predictModelAction() {
  uint32_t inferenceStartUs = micros();
  float *currentValues = modelValuesA;
  float *nextValues = modelValuesB;

  for (int index = 0; index < MODEL_INPUT_SIZE; ++index) {
    currentValues[index] = modelState[index];
  }

  for (int layerIndex = 0; layerIndex < MODEL_NUM_LAYERS; ++layerIndex) {
    runModelLayer(currentValues, nextValues, layerIndex);
    int outputSize = LAYER_OUTPUT_SIZES[layerIndex];

    if (layerIndex < MODEL_NUM_LAYERS - 1) {
      applyModelActivation(nextValues, outputSize, layerIndex);
    }

    float *temporary = currentValues;
    currentValues = nextValues;
    nextValues = temporary;
  }

  uint8_t bestAction = 0;
  for (int action = 0; action < MODEL_OUTPUT_SIZE; ++action) {
    modelQValues[action] = currentValues[action];
    if (modelQValues[action] > modelQValues[bestAction]) {
      bestAction = action;
    }
  }

  modelInferenceTimeUs = micros() - inferenceStartUs;
  return bestAction;
}

void updateModelState() {
  float metresPerStep = RAIL_LENGTH_METRES / static_cast<float>(railLengthSteps);
  modelState[0] =
      (static_cast<float>(currentPositionSteps) -
       0.5f * static_cast<float>(railLengthSteps)) *
      metresPerStep;
  modelState[1] = static_cast<float>(currentPolicyStepRatePps) * metresPerStep;
  modelState[2] = wrapRadians(MODEL_PENDULUM_SIGN * pendulumAngleRadians);
  modelState[3] =
      MODEL_PENDULUM_SIGN * pendulumAngularVelocityRadPerSec;
}

void updateDqnAction() {
  uint32_t safetyMargin = railLengthSteps / LIMIT_POSITION_MARGIN_DIVISOR;
  uint32_t brakingDistance = policyBrakingDistanceSteps();
  uint32_t middle = railLengthSteps / 2;

  uint32_t closeReversalPosition = safetyMargin + brakingDistance;
  if (closeReversalPosition > middle) {
    closeReversalPosition = middle;
  }

  uint32_t farReversalPosition =
      railLengthSteps > safetyMargin + brakingDistance
          ? railLengthSteps - safetyMargin - brakingDistance
          : middle;
  if (farReversalPosition < middle) {
    farReversalPosition = middle;
  }

  if (railSafetyOverrideDirection < 0) {
    if (currentPositionSteps <=
        static_cast<int32_t>(railLengthSteps - 2 * safetyMargin)) {
      railSafetyOverrideDirection = 0;
    } else {
      setControllerAction(0, "leaving far safety zone");
      return;
    }
  } else if (railSafetyOverrideDirection > 0) {
    if (currentPositionSteps >= static_cast<int32_t>(2 * safetyMargin)) {
      railSafetyOverrideDirection = 0;
    } else {
      setControllerAction(2, "leaving close safety zone");
      return;
    }
  }

  if (currentPolicyStepRatePps > 0 &&
      currentPositionSteps >= static_cast<int32_t>(farReversalPosition)) {
    railSafetyOverrideDirection = -1;
    setControllerAction(0, "far software boundary");
    return;
  }

  if (currentPolicyStepRatePps < 0 &&
      currentPositionSteps <= static_cast<int32_t>(closeReversalPosition)) {
    railSafetyOverrideDirection = 1;
    setControllerAction(2, "close software boundary");
    return;
  }

  updateModelState();
  uint32_t now = millis();
  if (lastDqnDecisionTimeMs == 0 ||
      now - lastDqnDecisionTimeMs >= DQN_DECISION_PERIOD_MS) {
    uint8_t predictedAction = predictModelAction();
    lastDqnDecisionTimeMs = now;

    if (!dqnHasDecision) {
      currentAction = predictedAction;
      dqnHasDecision = true;
      if (ENABLE_RUNTIME_STATUS) {
        printActionChange("first DQN decision");
      }
    } else if (predictedAction != currentAction) {
      currentAction = predictedAction;
      ++dqnActionSwitchCount;
    }
  }
}

void printStatus() {
  if (!ENABLE_RUNTIME_STATUS) {
    return;
  }

  uint32_t now = millis();
  if (now - lastStatusTime < STATUS_PERIOD_MS) {
    return;
  }

  lastStatusTime = now;
  Serial.print("[");
  Serial.print(now);
  Serial.print(" ms] Status | action: ");
  Serial.print(currentAction);
  Serial.print(" (");
  Serial.print(actionName(currentAction));
  Serial.print(")");
  Serial.print(" | Estimated position: ");
  Serial.print(currentPositionSteps);
  Serial.print(" / ");
  Serial.print(railLengthSteps);
  Serial.print(" | rate: ");
  Serial.print(currentPolicyStepRatePps);
  Serial.print(" -> ");
  Serial.print(targetRateForAction(currentAction));
  Serial.print(" pulses/s | braking: ");
  Serial.print(policyBrakingDistanceSteps());
  Serial.print(" steps | close: ");
  Serial.print(digitalRead(CLOSE_LIMIT_PIN) == LIMIT_TRIGGERED
                   ? "TRIGGERED"
                   : "OK");
  Serial.print(" | far: ");
  Serial.print(digitalRead(FAR_LIMIT_PIN) == LIMIT_TRIGGERED
                   ? "TRIGGERED"
                   : "OK");
  Serial.print(" | theta: ");
  Serial.print(pendulumAngleRadians * 180.0f / PI_F, 2);
  Serial.print(" deg | theta_dot: ");
  Serial.print(pendulumAngularVelocityRadPerSec, 3);
  Serial.print(" | mode: ");
  Serial.print("DQN");
  Serial.print(" | state: [");
  for (int index = 0; index < MODEL_INPUT_SIZE; ++index) {
    if (index > 0) {
      Serial.print(", ");
    }
    Serial.print(modelState[index], 3);
  }
  Serial.print("] | Q: [");
  for (int action = 0; action < MODEL_OUTPUT_SIZE; ++action) {
    if (action > 0) {
      Serial.print(", ");
    }
    Serial.print(modelQValues[action], 3);
  }
  Serial.print("] | inference: ");
  Serial.print(modelInferenceTimeUs);
  Serial.print(" us | switches: ");
  Serial.print(dqnActionSwitchCount);

  Serial.println();
}

void setup() {
  Serial.begin(115200);

  pinMode(ENABLE_PIN, OUTPUT);
  pinMode(DIRECTION_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(CLOSE_LIMIT_PIN, INPUT_PULLUP);
  pinMode(FAR_LIMIT_PIN, INPUT_PULLUP);

  disableMotor();
  digitalWrite(DIRECTION_PIN, CLOSE_DIRECTION);

  Serial.println("Keep the pendulum still at the bottom during startup.");
  if (!initializePendulumEncoder()) {
    stopWithError("AS5600 initialization or magnet check failed.");
    return;
  }

  if (!calibrateRail()) {
    return;
  }

  if (!updatePendulumState()) {
    stopWithError("AS5600 angle read failed after rail calibration.");
    return;
  }

  stepTimer = timerBegin(1000000);
  if (stepTimer == nullptr) {
    stopWithError("could not initialize the STEP pulse timer.");
    return;
  }
  timerAttachInterrupt(stepTimer, &onStepTimer);
  timerAlarm(stepTimer, STEP_TIMER_TICK_US, true, 0);

  controllerReady = true;
  lastControlTimeMs = millis() - CONTROL_PERIOD_MS;
  lastStatusTime = millis();
  currentAction = 1;
  lastDqnDecisionTimeMs = 0;
  dqnActionSwitchCount = 0;
  dqnHasDecision = false;
  railSafetyOverrideDirection = 0;
  Serial.println("Full DQN swing-up and balance control started.");
  Serial.println("Actions: 0 = close, 1 = stop, 2 = far");
  Serial.print("DQN network: ");
  Serial.print(MODEL_INPUT_SIZE);
  Serial.print(" inputs, ");
  Serial.print(MODEL_NUM_LAYERS);
  Serial.print(" layers, ");
  Serial.print(MODEL_OUTPUT_SIZE);
  Serial.println(" actions");
  Serial.print("Policy maximum rate: ");
  Serial.print(MAX_POLICY_STEP_RATE_PPS);
  Serial.print(" pulses/s | acceleration: ");
  Serial.print(POLICY_ACCELERATION_PPS2);
  Serial.println(" pulses/s^2");
  Serial.print("Software safety margin: ");
  Serial.print(railLengthSteps / LIMIT_POSITION_MARGIN_DIVISOR);
  Serial.println(" steps from each endpoint");
  Serial.print("DQN decision period: ");
  Serial.print(DQN_DECISION_PERIOD_MS);
  Serial.println(" ms | action selection: highest Q value");
  Serial.print("Runtime status logging: ");
  Serial.println(ENABLE_RUNTIME_STATUS ? "ENABLED" : "DISABLED");
  Serial.println("The DQN controls the pendulum from the bottom position.");
}


void finishPositionLog() {
  while (currentPolicyStepRatePps != 0) {
    if (timerLimitHalt || anyLimitIsTriggered()) {
      break;
    }
    currentPolicyStepRatePps = moveRateToward(currentPolicyStepRatePps, 0);
    setTimerStepRate(currentPolicyStepRatePps);
    delay(CONTROL_PERIOD_MS);
  }
  controllerReady = false;
  disableMotor();
  sendPositionLog();
}

void loop() {
  if (!controllerReady) {
    disableMotor();
    delay(100);
    return;
  }

  if (positionLogStarted && micros() - positionLogStartUs >= LOG_DURATION_US) {
    finishPositionLog();
    return;
  }

  uint32_t now = millis();
  if (now - lastControlTimeMs < CONTROL_PERIOD_MS) {
    delay(1);
    return;
  }
  lastControlTimeMs = now;

  if (!updatePendulumState()) {
    stopWithError("AS5600 angle read failed.");
    return;
  }

  // Start at the first post-calibration control sample, immediately before
  // the first DQN decision. Calibration and startup messages are excluded.
  uint32_t sampleUs = previousEncoderSampleTimeUs;
  if (!positionLogStarted) {
    positionLogStartUs = sampleUs;
    positionLogStarted = true;
  }
  updateDqnAction();
  recordPositionSample(sampleUs);

  if (!runPolicyAction(currentAction)) {
    return;
  }

  printStatus();
}
