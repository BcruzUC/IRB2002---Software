

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

  SetTunings(4, 0, 0);
  SetSampleTime(100);
}

void avanzar(int cant);
void volver();

// CONTROL DESDE INTERNET

unsigned long lastTime;
double Input, Output, Setpoint;
double errSum, lastInput;
double kp, ki, kd;
int SampleTime = 1000;

/////////////////////////////

int Step = 1;
int minSet = 120;
int maxSet = 240;

char instruccion;

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


    if (Serial.available() > 0){
      instruccion = Serial.read();
    }

    
     Input = analogRead(F1);

    if (instruccion == 'd' and Input > minSet){
      Compute_error();
      avanzar(Output);
      if ( Output < 20){
        Detener();
        instruccion = 'j';
      }
    }
    else if (instruccion == 's' && Input < maxSet){
      while(analogRead(F1) <= maxSet){
        volver(200);
      }
      if (Input <= maxSet){
      Detener();
      instruccion = 'j';
      }
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

void Detener(){
  analogWrite(ENA, 0);
}

int promedio(){
  int prom = 0; 
  for (int i; i<=5; i++){
    int dato = analogRead(F1);
    prom = prom + dato;
  }
  int promedio = prom / 6;
  return promedio;
}

void Compute_error()
{
  unsigned long now = millis();
  int timeChange = (now - lastTime);
    if (timeChange>=SampleTime)
       {
    // Calcula todas las variables de errores.
        double error = minSet - Input;
        errSum += error;
        double dInput = (Input - lastInput);
    
    // Calculamos la función de salida del PID.
          Output = abs(kp * error + ki * errSum - kd * dInput);
    // Guardamos el valor de algunas variables para el próximo ciclo de cálculo.
          lastInput = Input;
          lastTime = now;
       }
}

void SetTunings(double Kp, double Ki, double Kd)
{
    double SampleTimeInSec = ((double)SampleTime)/1000;
    kp = Kp;
    ki = Ki * SampleTimeInSec;
    kd = Kd / SampleTimeInSec;
}

void SetSampleTime(int NewSampleTime)
{
  if (NewSampleTime > 0)
     {
      double ratio  = (double)NewSampleTime / (double)SampleTime;
      ki *= ratio;
      kd /= ratio;
      SampleTime = (unsigned long)NewSampleTime;
   }
}
