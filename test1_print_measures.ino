

#include "EMGFilters.h"

#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#define SensorInput1 A0 // dedo inice
#define IN1 8 // lado izquierdo
#define IN2 7 // lado derecho
#define ENA 5 // enable A

#define F1 A1

int baseline = 17;

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
  
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);   //EnA

  pinMode(F1, INPUT);
}

void avanzar(int cant);
void volver();


int Step = 1;
int minSet = 170;
int maxSet = 235;

char instruccion;
int e0, ea0, ed0, ei0, e_tmp0 = 0;
int newposition, oldposition;
int kp, ki, kd;
long Time, t_ant;

void loop() {

    int data = analogRead(SensorInput1);
    // filter processing
    int dataAfterFilter =  myFilter.update(data);
    // Any value below the `baseline` value will be treated as zero
    if (dataAfterFilter < baseline and dataAfterFilter > (baseline * -1)) {
        dataAfterFilter = 0;
    }
    // You may plot the data using Arduino SerialPlotter.
    Serial.println(dataAfterFilter);


    newposition = analogRead(F1);

    if (Serial.available() > 0){
      instruccion = Serial.read();
    }

    if (instruccion == 'd' and newposition > minSet){
      oldposition = newposition;

      ea0= e0;
      e0= minSet - newposition;
  
      kp= 6 ;
      ki= 0 ;
      kd= 1 ;
  
      Time = micros();
      float ed0 = (e0-ea0)/(Time - t_ant)*100000;
      
      //// INTEGRATIVO
      e_tmp0 = ea0 + e_tmp0;
  
      float ei0 = (e_tmp0 + e0)*(Time-t_ant)/100000;
      t_ant = Time;
      
      //-----------------------------------
      // Salida al Motor
      //motorout1 = escalon;
      int motorout = (kp*e0 + kd*ed0 + ki*ei0) * 2;
      avanzar(motorout);
    }
    else if (instruccion == 's' && newposition < maxSet){
      while(analogRead(F1) <= maxSet){
        volver(200);
      }
      delay(400);
      analogWrite(ENA, 0);
    }
    // Serial.println(newposition);
    
    
}

void avanzar(int cant){
  analogWrite(ENA, cant);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN1, LOW);
}

void volver(int cant){
  analogWrite(ENA, cant);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
}
