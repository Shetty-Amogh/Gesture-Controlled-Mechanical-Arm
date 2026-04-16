#include <string.h>
#include <Servo.h>

const int servo_nos = 4;
const int servoPins[servo_nos] = {5, 6, 10, 11};

Servo servos[servo_nos];

float values[5];      
char buffer[100];      
int valueCount = 0;

int angles[5] = {0,0,0,0,0};

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < servo_nos; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].write(90);   // start position
  }
}

void loop() {
  if (Serial.available()) {

    //reading input values

    int len = Serial.readBytesUntil('!', buffer, sizeof(buffer)-1);
    buffer[len] = '\0'; 

    // putting values in values array

    char *ptr = strtok(buffer, ",");
    valueCount = 0;

    while (ptr != NULL && valueCount < 5) {
      values[valueCount] = atof(ptr);
      valueCount++;
      ptr = strtok(NULL, ",");
    }

    // main logic if inputted values are of length 5

    if(valueCount == 5){
      for (int i = 0; i < valueCount; i++) {
        angles[i] = int_to_deg(values[i]);
      }
      for (int i = 0; i < valueCount; i++){
        servos[i].write(angles[i]);
      }
    }
  }
}

int int_to_deg(float a){
  return ((int)(a * 90.0));
}
