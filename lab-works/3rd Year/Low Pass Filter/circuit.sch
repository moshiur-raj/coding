<Qucs Schematic 0.0.24>
<Properties>
  <View=0,0,1255,800,1,0,0>
  <Grid=10,10,1>
  <DataSet=circuit.dat>
  <DataDisplay=circuit.dpl>
  <OpenDisplay=1>
  <Script=circuit.m>
  <RunScript=0>
  <showFrame=0>
  <FrameText0=Title>
  <FrameText1=Drawn By:>
  <FrameText2=Date:>
  <FrameText3=Revision:>
</Properties>
<Symbol>
</Symbol>
<Components>
  <VProbe Pr1 1 1180 330 -16 28 0 3>
  <GND * 1 560 400 0 0 0 0>
  <R_SPICE R1 1 1060 330 15 -26 0 1 "159" 1 "" 0 "" 0 "" 0 "" 0>
  <C C1 1 500 330 17 -26 0 1 "20nF" 1 "" 0 "neutral" 0>
  <C C2 1 920 330 17 -26 0 1 "20nF" 1 "" 0 "neutral" 0>
  <L L1 1 570 260 -26 10 0 0 "0.5mH" 1 "" 0>
  <L L2 1 850 260 -26 10 0 0 "0.5mH" 1 "" 0>
  <R R2 1 710 260 -26 15 0 0 "21 Ohm" 1 "26.85" 0 "0.0" 0 "0.0" 0 "26.85" 0 "european" 0>
  <.AC AC1 1 520 640 0 41 0 0 "log" 1 "10Hz" 1 "700kHz" 1 "1000" 1 "no" 0>
  <Vac V1 1 380 330 18 -26 0 1 "0.707106781 V" 1 "1 kHz" 0 "0" 0 "0" 0>
</Components>
<Wires>
  <380 260 380 300 "" 0 0 0 "">
  <380 260 500 260 "" 0 0 0 "">
  <920 260 920 300 "" 0 0 0 "">
  <920 360 920 400 "" 0 0 0 "">
  <380 400 500 400 "" 0 0 0 "">
  <380 360 380 400 "" 0 0 0 "">
  <500 260 500 300 "" 0 0 0 "">
  <500 400 560 400 "" 0 0 0 "">
  <500 360 500 400 "" 0 0 0 "">
  <920 260 1060 260 "" 0 0 0 "">
  <1060 260 1060 280 "" 0 0 0 "">
  <1060 360 1060 380 "" 0 0 0 "">
  <920 400 1060 400 "" 0 0 0 "">
  <1160 280 1160 320 "" 0 0 0 "">
  <1060 280 1060 300 "" 0 0 0 "">
  <1060 280 1160 280 "" 0 0 0 "">
  <1160 340 1160 380 "" 0 0 0 "">
  <1060 380 1060 400 "" 0 0 0 "">
  <1060 380 1160 380 "" 0 0 0 "">
  <560 400 920 400 "" 0 0 0 "">
  <500 260 540 260 "" 0 0 0 "">
  <880 260 920 260 "" 0 0 0 "">
  <600 260 680 260 "" 0 0 0 "">
  <740 260 820 260 "" 0 0 0 "">
</Wires>
<Diagrams>
</Diagrams>
<Paintings>
</Paintings>