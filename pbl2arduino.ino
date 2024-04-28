#define red_pin 2
#define yellow_pin 3
#define green_pin 4

void setup() {
  pinMode(red_pin, OUTPUT);
  pinMode(yellow_pin, OUTPUT);
  pinMode(green_pin, OUTPUT);

  // Initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming byte:
    int input = Serial.read() - '0'; // Convert ASCII to integer

    // Turn off all lights by default
    digitalWrite(red_pin, LOW);
    digitalWrite(yellow_pin, LOW);
    digitalWrite(green_pin, LOW);

    // Turn on corresponding light based on input
    if (input == 0) {
      digitalWrite(red_pin, HIGH);
    } else if (input == 1) {
      digitalWrite(yellow_pin, HIGH);
    } else if (input == 2) {
      digitalWrite(green_pin, HIGH);
    }

    // Wait for a short time before checking again
    delay(100);
  }
}
