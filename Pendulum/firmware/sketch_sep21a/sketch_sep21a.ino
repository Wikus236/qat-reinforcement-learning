#include <Wire.h>
#include <soc/gpio_struct.h>

// Linear pendulum motor controller using an MKS TMC2160-OC driver and AS5600.

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
// keeps the proven 1000-pulse/s rate; a hardware timer maintains policy pulses.
constexpr uint32_t CONTROL_PERIOD_MS = 10;
constexpr int32_t MAX_POLICY_STEP_RATE_PPS = 10000;
constexpr int32_t POLICY_ACCELERATION_PPS2 = 80000;
constexpr int32_t POLICY_RATE_CHANGE_PER_PERIOD =
    POLICY_ACCELERATION_PPS2 * CONTROL_PERIOD_MS / 1000;
static_assert(POLICY_ACCELERATION_PPS2 > 0,
              "Policy acceleration must be greater than zero");
static_assert(POLICY_RATE_CHANGE_PER_PERIOD > 0,
              "Policy acceleration is too small for the control period");
constexpr uint32_t STEP_TIMER_TICK_US = 50;
constexpr uint32_t STEP_TIMER_TICKS_PER_SECOND = 1000000 / STEP_TIMER_TICK_US;
constexpr uint32_t STATUS_PERIOD_MS = 1000;
// Long Serial status lines delay control decisions while timer pulses continue.
constexpr bool ENABLE_RUNTIME_STATUS = false;
constexpr uint32_t SWING_START_KICK_MS = 300;
constexpr float SWING_VELOCITY_DEADBAND_RAD_PER_SEC = 0.15f;
constexpr float PENDULUM_LENGTH_METRES = 0.18f;
constexpr float GRAVITY_METRES_PER_SECOND_SQUARED = 9.81f;
// The real mechanism loses energy through friction and motor reversals, so
// aim slightly above the ideal upright energy.
constexpr float SWING_TARGET_ENERGY_MULTIPLIER = 1.10f;
constexpr float BALANCE_CAPTURE_ANGLE_RADIANS = 15.0f * PI_F / 180.0f;
constexpr float BALANCE_FALLBACK_ANGLE_RADIANS = 35.0f * PI_F / 180.0f;
constexpr float BALANCE_CAPTURE_MAX_ANGULAR_VELOCITY = 3.5f;
constexpr float BALANCE_ANGLE_GAIN = 1.0f;
constexpr float BALANCE_ANGULAR_VELOCITY_GAIN = 0.15f;
constexpr float BALANCE_POSITION_GAIN = 0.04f;
constexpr float BALANCE_CART_VELOCITY_GAIN = 0.08f;
constexpr float BALANCE_ACTION_DEADBAND = 0.002f;
constexpr float RAIL_LENGTH_METRES = 0.8f;
constexpr float CONTROL_PENDULUM_SIGN = 1.0f;

// Change this to -1 if the pendulum amplitude consistently decreases. The
// correct sign depends on the encoder orientation and which rail end is FAR.
constexpr int8_t SWING_PHASE_DIRECTION = 1;

uint32_t railLengthSteps = 0;
volatile int32_t currentPositionSteps = 0;
bool controllerReady = false;
uint32_t lastStatusTime = 0;
uint8_t currentAction = 2;
int32_t currentPolicyStepRatePps = 0;
hw_timer_t *stepTimer = nullptr;
portMUX_TYPE stepTimerMux = portMUX_INITIALIZER_UNLOCKED;
volatile int32_t timerStepRatePps = 0;
volatile uint32_t timerStepPhase = 0;
volatile bool timerStepHigh = false;
volatile bool timerLimitHalt = false;
uint8_t timerDirection = CLOSE_DIRECTION;
uint32_t lastControlTimeMs = 0;
float encoderBottomReferenceRadians = 0.0f;
float pendulumAngleRadians = -PI_F;
float pendulumAngularVelocityRadPerSec = 0.0f;
float previousPendulumAngleRadians = -PI_F;
uint32_t previousEncoderSampleTimeUs = 0;
uint32_t swingStartTimeMs = 0;
bool balanceControlActive = false;
int8_t railSafetyOverrideDirection = 0;
float balanceSignal = 0.0f;
float swingEnergyError = 0.0f;
uint32_t balanceActionSwitchCount = 0;

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
  float newPendulumAngle = wrapRadians(angleFromBottom - PI_F);

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
    // An open normally-closed switch immediately blocks the next pulse.
    if ((GPIO.in1.data & ((1UL << (CLOSE_LIMIT_PIN - 32)) |
                            (1UL << (FAR_LIMIT_PIN - 32)))) != 0) {
      timerStepRatePps = 0;
      timerStepPhase = 0;
      timerLimitHalt = true;
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
  currentAction = 2;
  swingStartTimeMs = millis();
  balanceControlActive = false;
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
  Serial.println("Carriage returned to the middle. Swing-up control resumed.");
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

void setSwingAction(uint8_t nextAction, const char *reason) {
  if (nextAction == currentAction) {
    return;
  }

  currentAction = nextAction;
  if (ENABLE_RUNTIME_STATUS) {
    printActionChange(reason);
  }
}

void updateBalanceAction() {
  float metresPerStep = RAIL_LENGTH_METRES / static_cast<float>(railLengthSteps);
  float cartPosition =
      (static_cast<float>(currentPositionSteps) -
       0.5f * static_cast<float>(railLengthSteps)) *
      metresPerStep;
  float cartVelocity =
      static_cast<float>(currentPolicyStepRatePps) * metresPerStep;
  float controlAngle =
      wrapRadians(CONTROL_PENDULUM_SIGN * pendulumAngleRadians);
  float controlAngularVelocity =
      CONTROL_PENDULUM_SIGN * pendulumAngularVelocityRadPerSec;

  balanceSignal =
      BALANCE_ANGLE_GAIN * controlAngle +
      BALANCE_ANGULAR_VELOCITY_GAIN * controlAngularVelocity +
      BALANCE_POSITION_GAIN * cartPosition +
      BALANCE_CART_VELOCITY_GAIN * cartVelocity;

  uint8_t nextAction = 1;
  if (balanceSignal > BALANCE_ACTION_DEADBAND) {
    nextAction = 2;
  } else if (balanceSignal < -BALANCE_ACTION_DEADBAND) {
    nextAction = 0;
  }

  if (nextAction != currentAction) {
    currentAction = nextAction;
    ++balanceActionSwitchCount;
  }
}

void updateSwingUpAction() {
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
      setSwingAction(0, "leaving far safety zone");
      return;
    }
  } else if (railSafetyOverrideDirection > 0) {
    if (currentPositionSteps >= static_cast<int32_t>(2 * safetyMargin)) {
      railSafetyOverrideDirection = 0;
    } else {
      setSwingAction(2, "leaving close safety zone");
      return;
    }
  }

  if (currentPolicyStepRatePps > 0 &&
      currentPositionSteps >= static_cast<int32_t>(farReversalPosition)) {
    railSafetyOverrideDirection = -1;
    setSwingAction(0, "far software boundary");
    return;
  }

  if (currentPolicyStepRatePps < 0 &&
      currentPositionSteps <= static_cast<int32_t>(closeReversalPosition)) {
    railSafetyOverrideDirection = 1;
    setSwingAction(2, "close software boundary");
    return;
  }

  if (!balanceControlActive &&
      fabsf(pendulumAngleRadians) <= BALANCE_CAPTURE_ANGLE_RADIANS &&
      fabsf(pendulumAngularVelocityRadPerSec) <=
          BALANCE_CAPTURE_MAX_ANGULAR_VELOCITY) {
    balanceControlActive = true;
    Serial.println(
        "Pendulum entered the capture region. Classical balance activated.");
  }

  if (balanceControlActive) {
    if (fabsf(pendulumAngleRadians) >= BALANCE_FALLBACK_ANGLE_RADIANS) {
      balanceControlActive = false;
      swingStartTimeMs = millis() - SWING_START_KICK_MS;
      Serial.println(
          "Balance controller lost upright. Swing-up control reactivated.");
      return;
    }

    updateBalanceAction();
    return;
  }

  if (millis() - swingStartTimeMs < SWING_START_KICK_MS) {
    setSwingAction(2, "initial kick");
    return;
  }

  if (fabsf(pendulumAngularVelocityRadPerSec) <
      SWING_VELOCITY_DEADBAND_RAD_PER_SEC) {
    return;
  }

  // Regulate the pendulum energy toward the upright energy. Unlike simple
  // phase pumping, this reverses the action when the pendulum has enough
  // energy and therefore prevents continuous full rotations.
  float tipVelocity =
      PENDULUM_LENGTH_METRES * pendulumAngularVelocityRadPerSec;
  float currentEnergy =
      0.5f * tipVelocity * tipVelocity +
      GRAVITY_METRES_PER_SECOND_SQUARED * PENDULUM_LENGTH_METRES *
          cosf(pendulumAngleRadians);
  float uprightEnergy =
      SWING_TARGET_ENERGY_MULTIPLIER *
      GRAVITY_METRES_PER_SECOND_SQUARED * PENDULUM_LENGTH_METRES;
  swingEnergyError = uprightEnergy - currentEnergy;

  float phaseSignal =
      swingEnergyError * pendulumAngularVelocityRadPerSec *
      cosf(pendulumAngleRadians);
  int8_t desiredDirection = phaseSignal > 0.0f ? -1 : 1;
  desiredDirection *= SWING_PHASE_DIRECTION;

  setSwingAction(desiredDirection > 0 ? 2 : 0, "pendulum phase");
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
  Serial.print(" rad/s | phase: ");
  Serial.print(
      swingEnergyError * pendulumAngularVelocityRadPerSec *
          cosf(pendulumAngleRadians),
      3);
  Serial.print(" | energy error: ");
  Serial.print(swingEnergyError, 3);
  Serial.print(" | mode: ");
  Serial.print(balanceControlActive ? "BALANCE" : "SWING");

  if (balanceControlActive) {
    Serial.print(" | balance signal: ");
    Serial.print(balanceSignal, 3);
    Serial.print(" | switches: ");
    Serial.print(balanceActionSwitchCount);
  }

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
  currentAction = 2;
  swingStartTimeMs = millis();
  balanceControlActive = false;
  balanceSignal = 0.0f;
  swingEnergyError = 0.0f;
  balanceActionSwitchCount = 0;
  railSafetyOverrideDirection = 0;
  Serial.println("Classical swing-up and balance control started.");
  Serial.println("STEP pulses: hardware timer");
  Serial.println("Actions: 0 = close, 1 = stop, 2 = far");
  Serial.print("Policy maximum rate: ");
  Serial.print(MAX_POLICY_STEP_RATE_PPS);
  Serial.print(" pulses/s | acceleration: ");
  Serial.print(POLICY_ACCELERATION_PPS2);
  Serial.println(" pulses/s^2");
  Serial.print("Software safety margin: ");
  Serial.print(railLengthSteps / LIMIT_POSITION_MARGIN_DIVISOR);
  Serial.println(" steps from each endpoint");
  Serial.print("Balance capture region: +/-");
  Serial.print(BALANCE_CAPTURE_ANGLE_RADIANS * 180.0f / PI_F, 1);
  Serial.print(" degrees at no more than ");
  Serial.print(BALANCE_CAPTURE_MAX_ANGULAR_VELOCITY, 1);
  Serial.println(" rad/s");
  Serial.print("Runtime status logging: ");
  Serial.println(ENABLE_RUNTIME_STATUS ? "ENABLED" : "DISABLED");
  if (ENABLE_RUNTIME_STATUS) {
    printActionChange("initial kick");
  }
}

void loop() {
  if (!controllerReady) {
    disableMotor();
    delay(100);
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

  updateSwingUpAction();

  if (!runPolicyAction(currentAction)) {
    return;
  }

  printStatus();
}
