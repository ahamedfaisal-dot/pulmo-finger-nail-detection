// Life Trace Edge Hub - 3D Printable Enclosure
// Compatible with OpenSCAD (https://openscad.org/)
// 
// This script generates a parameterized 3D model for your project.
// Render it in OpenSCAD and export as STL for your 3D printer.

/* [Dimensions] */
box_width = 120;
box_length = 90;
box_height = 50;
wall_thickness = 2;

/* [Finger Cradle] */
finger_radius = 12;
finger_length = 50;

/* [Raspberry Pi Mount] */
pi_width = 85;
pi_length = 56;
mount_height = 5;

module base_box() {
    difference() {
        // Outer Shell
        cube([box_width, box_length, box_height], center=true);
        // Inner Cavity
        translate([0, 0, wall_thickness])
            cube([box_width - wall_thickness*2, box_length - wall_thickness*2, box_height], center=true);
    }
}

module finger_cradle() {
    // A rounded cut-out on the top surface where the user places their finger
    translate([0, box_length/2 - finger_length/2 - 10, box_height/2])
        rotate([90, 0, 0])
            cylinder(r=finger_radius, h=finger_length, center=true, $fn=50);
}

module sensor_cutouts() {
    // Hole for Raspberry Pi Camera (pointing at the finger)
    // Assuming camera is mounted above or looking directly at the finger
    translate([0, box_length/2 - 25, box_height/2 - finger_radius])
        cylinder(r=5, h=10, center=true, $fn=30);
        
    // Hole for MAX30102 Heart Rate Sensor (bottom of the finger cradle)
    translate([-10, box_length/2 - 30, box_height/2 - finger_radius])
        cylinder(r=4, h=10, center=true, $fn=30);
        
    // Hole for MLX90614 Temperature Sensor
    translate([10, box_length/2 - 30, box_height/2 - finger_radius])
        cylinder(r=4, h=10, center=true, $fn=30);
}

module pi_mounts() {
    // Simple mounting standoffs for Raspberry Pi 4
    mount_x = pi_width/2 - 3.5;
    mount_y = pi_length/2 - 3.5;
    
    translate([0, -10, -box_height/2 + wall_thickness/2 + mount_height/2]) {
        for (x = [-mount_x, mount_x]) {
            for (y = [-mount_y, mount_y]) {
                translate([x, y, 0])
                    difference() {
                        cylinder(r=3, h=mount_height, center=true, $fn=20);
                        cylinder(r=1.2, h=mount_height+1, center=true, $fn=20); // Screw hole
                    }
            }
        }
    }
}

// Assemble the final case
difference() {
    union() {
        base_box();
        pi_mounts();
    }
    // Subtract the finger cradle and sensor holes from the top
    finger_cradle();
    sensor_cutouts();
    
    // Add port cutouts for Raspberry Pi (Power/HDMI)
    translate([box_width/2, -10, -box_height/2 + 10])
        cube([10, 40, 15], center=true);
}
