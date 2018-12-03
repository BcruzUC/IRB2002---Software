
#define IN1 8 // lado izquierdo
#define IN2 7 // lado derecho
#define ENA 5 // enable A

#define Fl A1
#define Fu A2



void setup()
{
  Serial.begin(115200);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);   //EnA

  pinMode(Fl, INPUT);
  pinMode(Fu, INPUT);

  SetTunings(10, 0, 2);
  SetSampleTime(100);
}



// CONTROL DESDE INTERNET

unsigned long lastTime;
double Input, Output, Setpoint;
double errSum, lastInput;
double kp, ki, kd;
int SampleTime = 1000;

/////////////////////////////


int Step = 1;
int minSet = 105;
int maxSet = 140;

char instruccion;
int e0, ea0, ed0, ei0, e_tmp0 = 0;
int newposition, oldposition;
//int kp, ki, kd;
long Time, t_ant;

char temp_inst;
int data_int;


void loop() {
    //-----------------------------------
    // Actualizando Informacion de los encoders
    newposition = promedio();

    if (split_data() > 0){
      instruccion = temp_inst;
      minSet = data_int;
    }

    while (instruccion == 'd' and newposition > minSet){
      Input = promedio();
      //int Force = analogRead(Fu);   //incluir en la tomada.. se corta cuando fuerza es mayor a X
      Compute_error();
      avanzar(Output);
      if ( Output < 30){
        Detener();
        instruccion = 'j';
      }
    }
    if (instruccion == 's' and Input <= maxSet){
      while(analogRead(Fl) <= maxSet){
        volver(200);
      }
      if (Input <= maxSet){
      Detener();
      instruccion = 'j';
      }
    }
    Serial.print("PROMEDIO: ");
    Serial.println(newposition);
    Serial.print("  ||  Error: ");
    Serial.print(Output);
    

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
  int ac_prom = 0; 
  for (int i=0; i<=5; i++){
    int dato = analogRead(Fl);
    ac_prom = ac_prom + dato;
  }
  int promedio = ac_prom / 6;
  return promedio;
}

void Compute_error()
{
  unsigned long now = millis();
  int timeChange = (now - lastTime);
    if (timeChange>=SampleTime)
       {
    // Calcula todas las variables de errores.
        double error = abs(minSet - Input);
        errSum += error;
        double dInput = abs(Input - lastInput);
    
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

int split_data(){
  if (Serial.available() > 0){
    char servo = Serial.read();
    if (servo == 's'){
      temp_inst = servo;
      return 1;
    }
    if(servo == 'd'){
      //here you could check the servo number
      String pos = Serial.readStringUntil(';');
      int int_pos = pos.toInt();
      temp_inst = servo;
      data_int = int_pos;
      return 1;
      }
  }
return 0;
}
  


