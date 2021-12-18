
yawRef = zeros(1,length(y_0848_world));
for i= 1:(length(yawRef)-1)
    yawRef(i) = atan2( y_0848_world(i+1)- y_0848_world(i), x_0848_world(i+1)-x_0848_world(i));
end
yawRef(end) = yawRef(end-1);
yawRef = rad2deg(yawRef)/10;
