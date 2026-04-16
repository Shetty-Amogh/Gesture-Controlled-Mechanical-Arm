#include <string.h>

float values[5];      // Float array for your data
float prev_values[] = {0,0,0,0,0};
char buffer[100];      // Serial buffer
int valueCount = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    int len = Serial.readBytesUntil('!', buffer, sizeof(buffer)-1);
    buffer[len] = '\0'; 

    char *ptr = strtok(buffer, ",");
    valueCount = 0;

    while (ptr != NULL && valueCount < 5) {
      values[valueCount] = atof(ptr);
      valueCount++;
      ptr = strtok(NULL, ",");
    }
    Serial.println(valueCount);
    if(valueCount == 5){
      for (int i = 0; i < valueCount; i++) {
        Serial.println(values[1]);
        Serial.println(prev_values[1]);
        if(values[1] < prev_values[1]){
          //finger closing code 
          Serial.println("Finger Closing ");
        }
        else if(values[1] > prev_values[1]){
          //finger opening code
          Serial.println("Finger Opening");
        }

        prev_values[i] = values[i];
      }
    }

    
  }
}
