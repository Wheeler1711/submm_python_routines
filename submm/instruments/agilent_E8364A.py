import pyvisa
import numpy as np
import time

class vna(object):

    def __init__(self,address = None):
        if not address:
            self.GPIBaddress = 'GPIB0::20::INSTR'
        else:
            self.GPIBaddress = address
        rm = pyvisa.ResourceManager("@py")
        self.obj = rm.open_resource(self.GPIBaddress)
        self.transfermode = 2 #%1 = asc, 2 = binary transfer
        self.obj.timeout = 30000 #30 seconds
        
        

        #gpibtimeout = 5;%second
        #gpibbuffersize = 100000;


    
    

    def SetStat(self,bon):
    #instrument control
    #set the state to 1=on or 0=off
    
        if bon:
            cmd_str = ':OUTP ON;'
            self.obj.write(cmd_str)
        else:
            cmd_str = ':OUTP OFF;'
            self.obj.write(cmd_str)
          
    
    def SetAttAuto(self, bon):
        
        if bon:
            cmd_str = 'SOUR:POW1:ATT:AUTO ON;'
            self.obj.write(cmd_str)
        else:
            cmd_str = 'SOUR:POW1:ATT:AUTO OFF;'
            self.obj.write(cmd_str)
        
    
    
    def SetAttenuation(self, att):

        cmd_str =  'SOUR:POW2:ATT %d;' % np.abs(np.round(att))
        self.obj.write(cmd_str)        
    
   
    #take snapshot
    def alysnapshot(self,fcenter, fspan, pow, filename = None, averfact = 1, points =1601, ifbw = 1000, Smn = "S21"):
      
        
        ff = 1 #1 = ascii read, 2 = binary transfer
        
        #get the measurement string and then select
        cmd_str = 'CALC:PAR:CAT?'
        answer = self.obj.query(cmd_str)   
        #MeaString = ['"' str(2:findstr(answer,',')-1) '"'];
        MeaString = answer.split(",")[0][1:]#str(2:findstr(answer,',')-1);
        #print(MeaString)
        cmd_str = 'CALC:PAR:SEL ' + MeaString +';'
        self.obj.write(cmd_str)

        self.obj.write('CALC:PAR:MOD ' +Smn +';' )
          
        

        N = points;
        cmd_str = 'SENS:SWE:POIN %d;' % points
        self.obj.write(cmd_str);
       
        
        #set sweeptime automatically
        self.obj.write('SENS:SWE:TIME:AUTO 1')
        
        #set bandwidth
        cmd_str = 'SENS:BWID %.2f HZ;' % ifbw
        self.obj.write(cmd_str);
  
        
        #set twait

        sweeptime = float(self.obj.query('SENS:SWE:TIME?'))
        twait = sweeptime*averfact * 1.02;

        
        #set power and frequency
        self.obj.write('SENS:SWE:TYPE LIN;')
        cmd_str = 'SOUR:POW1 %.2f;' % pow
        self.obj.write(cmd_str)
        cmd_str = 'SENS:FREQ:CENT %.9f GHz;' % fcenter
        self.obj.write(cmd_str)
        cmd_str = 'SENS:FREQ:SPAN %.9f GHz' % fspan
        self.obj.write(cmd_str)
        
        if self.transfermode == 1:
            self.obj.write('FORM:DATA ASCii,0')
        else: #(obj.transfermode==2)
            self.obj.write('FORM:DATA REAL,32')
            self.obj.write('FORM:BORD SWAP')
        
        
        if averfact > 0:
            cmd_str = 'SENS:AVER:COUN %d;' % averfact
            self.obj.write(cmd_str)
            self.obj.write('SENS:AVER ON;')
            
            cmd_str = 'SENS:SWE:GRO:COUN %d;' % averfact
            self.obj.write(cmd_str)
            self.obj.write('INIT:CONT ON')
            
            answer = self.obj.query('SENS:SWE:MODE GRO;*OPC?')
            #print(answer)
            #sleep(twait)
            #fscanf(obj);
        
        
        self.obj.write('DISP:WIND:TRAC:Y:AUTO') #Autoscale display
        
        if self.transfermode == 1: #asc #probably doesn't work
            k = self.obj.query('CALC:DATA? SDATA;')
            print(k)
            #km = sscanf(k, '%g,%g,', [2 Inf])';
        else: #bin
            #self.obj.write('CALC:DATA? SDATA;')
            #header = self.obj.read(
            answer = np.asarray(self.obj.query_binary_values('CALC:DATA? SDATA;'),dtype =  np.double)
            #print(answer)
            
  
                  
        
        fcenter = float(self.obj.query('SENS:FREQ:CENT?;'))#in Hz, %E8364A
        #fcenter = fscanf(obj, '%g');
        fspan = float(self.obj.query('SENS:FREQ:SPAN?;'))#in Hz, %E8364A
        #fspan = fscanf(obj, '%g');
        f = (np.linspace(-(N-1)/2,(N-1)/2,N)*fspan/(N-1) + fcenter)/1e9  #in GHz
        z = answer[::2] + 1j * answer[1::2]
        
        if filename is not None:
            with open(filename+".csv",'w') as file:
                file.write("center: "+str(fcenter)+" span: "+str(fspan) +
                " power: "+str(pow)+ " average: "+str(averfact)+" ifbw: " +str(ifbw)+ " Smn: " +Smn +"\n")
                for f_value,real,imag in list(zip(f,np.real(z),np.imag(z))):
                    file.write(F"{f_value},{real},{imag}\n")
        
            
        return f, z
        
    

    #take survey
    def survey(self, fwinstart, fwinend, fwinsize, pow, averfact = 1, ifbw = 1000,Smn = 'S21',filename = None):
       
        #display(nargin)
        #fix number of points at 1601
        N = 1601;
    
        fwinres = fwinsize/(N-1) #frquency resolution
        
        f = np.asarray(())
        z = np.asarray((),dtype = complex)
        for fwincenter in np.arange(fwinstart,fwinend,fwinsize):
            #fwin = [-800:1:800]' * fwinres + fwincenter;
            print(fwincenter,fwinsize);
             
            [fwin, zwin] = self.alysnapshot(fwincenter, fwinsize, pow, averfact= averfact, points=N, ifbw=ifbw,Smn=Smn);
         
            f = np.append(f,fwin[:-1])#(1:N-1)];
            z = np.append(z,zwin[:-1])#(1:N-1)];
            #drawnow;
        
        
        f = np.append(f, fwin[-1])
        z = np.append(z, zwin[-1])
        
        if filename is not None:
            with open(filename+".csv",'w') as file:
                file.write("start: "+str(fwinstart)+" stop: "+str(fwinend)+" window_span: " +str(fwinsize)+
                " power: "+str(pow)+ " average: "+str(averfact)+" ifbw: " +str(ifbw)+ " Smn: " +Smn +"\n")
                for f_value,real,imag in list(zip(f,np.real(z),np.imag(z))):
                    file.write(F"{f_value},{real},{imag}\n")
        
        return f,z
    
    
'''
    %take snapshot
    function [f,z,obj] = readtrace(obj)
        if(obj.gpibaddress <=0)
            return;
        
        
        ff = 1; % 1 = ascii read, 2 = binary transfer
        
        fprintf(obj, 'SENS:SWE:POIN?');
        N = fscanf(obj,'%d');
        
        if(obj.transfermode==1)
            fprintf(obj, 'FORM:DATA ASCii,0');
        elseif(obj.transfermode==2)
            fprintf(obj, 'FORM:DATA REAL,32');
            fprintf(obj, 'FORM:BORD SWAP');
        
        
        if(obj.transfermode == 1) %asc
            fprintf(obj, 'CALC:DATA? SDATA;');
            k=fscanf(obj);
            km = sscanf(k, '%g,%g,', [2 Inf])';
        elseif(obj.transfermode==2) %bin
            fprintf(obj, 'CALC:DATA? SDATA;');
            %#<num_digits><byte_count>data<NL><End>
            
            %header = fscanf(obj, '%c', 1);
            header = fread(obj,'char',1);
            
            %num_digits = str2num(fscanf(obj, '%c', 1));
            num_digits = str2num(fread(obj,'char',1));
            
            %nbyte = str2num(fscanf(obj, '%c', num_digits));
            nbyte = str2num(fread(obj,'char', num_digits));
            
            %km =fread(obj, [2,nbyte/8],'float32');
            mm = fread(obj,'single',nbyte);
            
            trailer = fread(obj);
            
            km(:,1) = double(mm(1:2:));
            km(:,2) = double(mm(2:2:));
        
        
        fprintf(obj, 'SENS:FREQ:CENT?;');%in Hz, %E8364A
        fcenter = fscanf(obj, '%g');
        fprintf(obj, 'SENS:FREQ:SPAN?;');%in Hz, %E8364A
        fspan = fscanf(obj, '%g');
        f = ([-(N-1)/2:1:(N-1)/2]'*fspan/(N-1) + fcenter)/1e9;  %in GHz
        z = km(:,1) + i * km(:,2);
        
        global showplot;
        if(showplot)
            figure(1);
            plot(f, 20*log10(abs(z)),'b.-');
        
    
    
    
    
    
    
    function [p, z] = alypowsweep(obj, powstart, powstop, CWfreq, filename, averfact, points, ifbw)
        
       if(obj.gpibaddress <=0)
            return;
       
       
       
       ff = 2;%file ascii or mat
       
       %set bandwidth
        if(nargin>7)
            fprintf(obj, 'SENS:BWID %.2f HZ;',ifbw);
        else
            ifbw = str2num(aly.ask('SENS:BWID?'));
        

        if(nargin>6)
            N = points;
            fprintf(obj, 'SENS:SWE:POIN %d;',points);
        else
            N = str2num(aly.ask('SENS:SWE:POIN?'));          
        
        
        
        if(nargin<6)
            averfact = 1;
        
        
        %get the measurement string and then select
        fprintf(obj, 'CALC:PAR:CAT?');
        str = fscanf(obj);
        MeaString = ['"' str(2:findstr(str,',')-1) '"'];
        fprintf(obj, ['CALC:PAR:SEL ' MeaString]);
        if (nargin>8)
            fprintf(obj, ['CALC:PAR:MOD ' Smn ]);
        
        
        %set twait
        global twait;
        fprintf(obj, 'SENS:SWE:TIME?');
        sweeptime = fscanf(obj, '%f');
        twait = sweeptime*averfact * 1.02;
        
        powcenter = (powstart+powstop)/2;
        powspan = (powstop-powstart);
        
        %set power and frequency
        fprintf(obj, 'SENS:SWE:TYPE POW;');
        %fprintf(obj, 'SENS:FOM:RANG:FREQ:CW %.9f GHz;', CWfreq);
        fprintf(obj, 'SENS:FREQ:CENT %.9f GHz;', CWfreq);
        fprintf(obj, 'SOUR:POW:CENT %.2f;', powcenter);
        fprintf(obj, 'SOUR:POW:SPAN %.2f;', powspan);
        
        if(obj.transfermode==1)
            fprintf(obj, 'FORM:DATA ASCii,0');
        elseif(obj.transfermode==2)
            fprintf(obj, 'FORM:DATA REAL,32');
            fprintf(obj, 'FORM:BORD SWAP');
        
        
        if(averfact > 0)
            fprintf(obj, 'SENS:AVER:COUN %d;', averfact);
            fprintf(obj, 'SENS:AVER ON;');
            
            fprintf(obj, 'SENS:SWE:GRO:COUN %d;', averfact);
            fprintf(obj, 'INIT:CONT ON');
            
            fprintf(obj, 'SENS:SWE:MODE GRO;*OPC?');
            pause(twait);
            fscanf(obj);
        
          
        
        fprintf(obj, 'DISP:WIND:TRAC:Y:AUTO'); %Autoscale display
        
        if(obj.transfermode == 1) %asc
            fprintf(obj, 'CALC:DATA? SDATA;');
            k=fscanf(obj);
            km = sscanf(k, '%g,%g,', [2 Inf])';
        elseif(obj.transfermode==2) %bin
            fprintf(obj, 'CALC:DATA? SDATA;');
            #<num_digits><byte_count>data<NL><End>
            
            %header = fscanf(obj, '%c', 1);
            header = fread(obj,'char',1);
            
            %num_digits = str2num(fscanf(obj, '%c', 1));
            num_digits = str2num(fread(obj,'char',1));
            
            %nbyte = str2num(fscanf(obj, '%c', num_digits));
            nbyte = str2num(fread(obj,'char', num_digits));
            
            %km =fread(obj, [2,nbyte/8],'float32');
            mm = fread(obj,'single',nbyte);
            
            trailer = fread(obj);
            
            km(:,1) = double(mm(1:2:));
            km(:,2) = double(mm(2:2:));
        
        
        
        fprintf(obj, 'SOUR:POW:CENT?;');%in Hz, %E8364A
        pcenter = fscanf(obj, '%g');
        
        fprintf(obj, 'SOUR:POW:SPAN?;');%in Hz, %E8364A
        pspan = fscanf(obj, '%g');
        p = ([-(N-1)/2:1:(N-1)/2]'*pspan/(N-1) + pcenter); 
        z = km(:,1) + i * km(:,2);
        

        if(showplot)
            figure(1);
            plot(p, 20*log10(abs(z)),'b.-');
        
        
        
        if(nargin > 4)
            save(filename, 'p','z','-mat');
       
'''

