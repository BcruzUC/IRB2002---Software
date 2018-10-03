

#include "EMGFilters.h"

#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#define SensorInput1 A0 // dedo inice
#define SensorInput2 A1 // dedo inice

#define CALIBRATE 0

int baseline = 100;
int baseline2 = 20;

EMGFilters myFilter;

SAMPLE_FREQUENCY sampleRate = SAMPLE_FREQ_1000HZ;

// Time interval for processing the input signal.
unsigned long long interval = 1000000ul / sampleRate;

// Set the frequency of power line hum to filter out.
//
// For countries with 60Hz power line, change to "NOTCH_FREQ_60HZ"
NOTCH_FREQUENCY humFreq = NOTCH_FREQ_50HZ;




void setup(void) {
  // put your setup code here, to run once:
  myFilter.init(sampleRate, humFreq, true, true, true);
  Serial.begin(115200);
  pinMode(13, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
    unsigned long long timeStamp = micros();

    int data = analogRead(SensorInput1);
    int data2 = analogRead(SensorInput2);
    // filter processing
    int dataAfterFilter =  myFilter.update(data);
    int dataAfterFilter2 = data2;//myFilter.update(data2);

    // Get envelope by squaring the input
    int envelope = sq(dataAfterFilter);
    int envelope2 = sq(dataAfterFilter2);

    if (CALIBRATE) {
        /*Serial.print("Squared Data: ");*/
        Serial.println(envelope);
    }
    else {
        // Any value below the `baseline` value will be treated as zero
        if (envelope < baseline) {
            dataAfterFilter = 0;
            envelope = 0;
        }
        if (envelope2 < baseline2) {
            dataAfterFilter2 = 0;
            envelope2 = 0;
        }
        if (envelope > 200){
          digitalWrite(13, HIGH);
        }
        else {
          digitalWrite(13, LOW);
        }
        // You may plot the data using Arduino SerialPlotter.
        Serial.println(envelope);
        //Serial.print(" || ");
        //Serial.println(envelope2);
    }
    unsigned long long timeElapsed = micros() - timeStamp;

}
