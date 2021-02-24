%MOUSECLICK Click the selected button of the mouse in the selected position
%   This moves mouse pointer to specified screen coordinates and presses
%   and releases the left mouse button in order to save 

% Import Java classes


% TCP/IP Server Sockets
t = tcpip('0.0.0.0', 44100, 'NetworkRole', 'Server');
fopen(t);
while(1)
    status = fgetl(t);
    if(strcmp(status, 'START'))
        % (100, 269) ACER
        % (103, 270) MSI
        mouse.mouseMove(100, 269); % control mouse pointer position
        mouse.mouseMove(100, 269); % repeating the operation to ensure pointer position moved
        mouse.mouseMove(100, 269);
        mouse.mouseMove(100, 269);
        pause(0.5)
        for i = 1:100
            pause(0.25);
            if(t.BytesAvailable > 5)
                fgetl(t);
                fprintf(t, 'NOK');
            end
            mouse.mousePress(InputEvent.BUTTON1_MASK); % left click press
            mouse.mouseRelease(InputEvent.BUTTON1_MASK); % left click release
        end
        
        line = fgetl(t);
        fprintf(t, 'OK');
        
%         folder = fgetl(t);
%         folder = folder(8:end);
%         mkdir(folder);
% %         movefile('C:\Users\danim\Documents\Images\*', folder) % ACER
%         movefile('D:\Datos\Images\*', folder) % MSI
    end
end