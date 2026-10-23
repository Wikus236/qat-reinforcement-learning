#include <Wire.h>

constexpr uint8_t ENCODER_SDA_PIN = 21;
constexpr uint8_t ENCODER_SCL_PIN = 22;

constexpr uint8_t AS5600_ADDRESS = 0x36;
constexpr uint8_t STATUS_REGISTER = 0x0B;
constexpr uint8_t RAW_ANGLE_REGISTER = 0x0C;
constexpr uint8_t AGC_REGISTER = 0x1A;
constexpr uint8_t MAGNITUDE_REGISTER = 0x1B;

constexpr uint8_t STATUS_MAGNET_DETECTED = 1 << 5;
constexpr uint8_t STATUS_MAGNET_TOO_WEAK = 1 << 4;
constexpr uint8_t STATUS_MAGNET_TOO_STRONG = 1 << 3;

constexpr uint32_t SERIAL_BAUD = 115200;
constexpr uint32_t SAMPLE_PERIOD_MS = 100;

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

  uint8_t highByte = Wire.read();
  uint8_t lowByte = Wire.read();
  value = (static_cast<uint16_t>(highByte & 0x0F) << 8) | lowByte;
  return true;
}

bool sensorIsConnected() {
  Wire.beginTransmission(AS5600_ADDRESS);
  return Wire.endTransmission() == 0;
}

const char *magnetStatusText(uint8_t status) {
  if ((status & STATUS_MAGNET_DETECTED) == 0) {
    return "NOT DETECTED";
  }
  if ((status & STATUS_MAGNET_TOO_WEAK) != 0) {
    return "TOO WEAK";
  }
  if ((status & STATUS_MAGNET_TOO_STRONG) != 0) {
    return "TOO STRONG";
  }
  return "OK";
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin(ENCODER_SDA_PIN, ENCODER_SCL_PIN);
  Wire.setClock(400000);
  delay(250);

  Serial.println("AS5600 sensor test");
  Serial.println("I2C pins: SDA = 21, SCL = 22");

  if (sensorIsConnected()) {
    Serial.println("AS5600 detected at I2C address 0x36.");
    Serial.println("raw_angle, degrees, wrapped_degrees, magnet, AGC, magnitude");
  } else {
    Serial.println("ERROR: AS5600 not detected at I2C address 0x36.");
    Serial.println("Check 3.3 V, GND, SDA and SCL wiring.");
  }
}

void loop() {
  uint8_t status = 0;
  uint8_t agc = 0;
  uint16_t rawAngle = 0;
  uint16_t magnitude = 0;

  bool readSucceeded =
      readRegister8(STATUS_REGISTER, status) &&
      readRegister12(RAW_ANGLE_REGISTER, rawAngle) &&
      readRegister8(AGC_REGISTER, agc) &&
      readRegister12(MAGNITUDE_REGISTER, magnitude);

  if (!readSucceeded) {
    Serial.println("ERROR: Could not read AS5600 registers.");
    delay(SAMPLE_PERIOD_MS);
    return;
  }

  float degrees = static_cast<float>(rawAngle) * 360.0f / 4096.0f;
  float wrappedDegrees = degrees > 180.0f ? degrees - 360.0f : degrees;

  Serial.print(rawAngle);
  Serial.print(", ");
  Serial.print(degrees, 2);
  Serial.print(", ");
  Serial.print(wrappedDegrees, 2);
  Serial.print(", ");
  Serial.print(magnetStatusText(status));
  Serial.print(", ");
  Serial.print(agc);
  Serial.print(", ");
  Serial.println(magnitude);

  delay(SAMPLE_PERIOD_MS);
}
