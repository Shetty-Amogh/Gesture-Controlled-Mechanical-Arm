#include <string.h>

float values[10];      // Float array for your data
char buffer[100];      // Serial buffer
int valueCount = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    int len = Serial.readBytesUntil('\n', buffer, sizeof(buffer)-1);
    buffer[len] = '\0'; 

    char *ptr = strtok(buffer, ",");
    valueCount = 0;

    while (ptr != NULL && valueCount < 10) {
      values[valueCount] = atof(ptr);  // Convert string to float
      valueCount++;
      ptr = strtok(NULL, ",");
    }

    Serial.print("Received ");
    Serial.print(valueCount);
    Serial.println(" float values:");
    
    for (int i = 0; i < valueCount; i++) {
      Serial.print("values[");
      Serial.print(i);
      Serial.print("] = ");
      Serial.println(values[i], 2);  // 2 decimal places
    }
  }
}