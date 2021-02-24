classdef MultiSpectralCamera
    % The current version of this implementation will carry out a
    % workaround in order to capture images. The workaround is based on the
    % use of a Robot on a separate computer. This robot will be implemented
    % in Python and will open a TCP Socket Server.
    
    properties (Access=private)
        socket;
        ip;
        port;
        mouse;
    end
    
    methods (Access=public)
        
        % Constructor
        function obj = MultiSpectralCamera(ip,port)
            import java.awt.Robot;
            import java.awt.event.*;
            obj.mouse = Robot;
            obj.ip = ip;
            obj.port = port;
            %obj.socket = tcpip(ip, port);
            %obj.socket.Terminator = 'LF';
            %fopen(obj.socket);
        end
        
        % Start capture
        function start_capture(obj)
            import java.awt.Robot;
            import java.awt.event.*;
            % We move the mouse to the desired position
            obj.mouse.mouseMove(100, 269); % control mouse pointer position
            obj.mouse.mouseMove(100, 269); % repeating the operation to ensure pointer position moved
            obj.mouse.mouseMove(100, 269);
            obj.mouse.mouseMove(100, 269);
            pause(1);
            
            % We capture 100 images pressing the button
            for i = 1:100
                pause(0.3);
                obj.mouse.mousePress(InputEvent.BUTTON1_MASK); % left click press
                obj.mouse.mouseRelease(InputEvent.BUTTON1_MASK); % left click release
            end
            
        end
        
        % Check if capture has finished
        function [out, err] = has_finished(obj)
            err = 0; % No error detected
            out = 1;
        end
        
    end
    
    
end