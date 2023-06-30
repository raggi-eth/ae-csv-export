// Get the current composition
var comp = app.project.activeItem;
if (!comp || !(comp instanceof CompItem)) {
    throw new Error("No active composition. Please select a composition and try again.");
}

// Show a dropdown of all layers in the composition
var wnd = new Window("dialog", "Select Layer");

var layerNames = [];
for (var i = 1; i <= comp.layers.length; i++) {
    layerNames.push(comp.layers[i].name);
}

var list = wnd.add("dropdownlist", undefined, layerNames);
list.selection = 0;

var btn = wnd.add("button", undefined, "OK");
btn.onClick = function() {
    wnd.close();
};
wnd.show();

// Get the selected layer
var layer = comp.layer(list.selection.text);
if (!layer) {
    throw new Error("No layer selected. Please try again.");
}

var data = ""; // Initialize the variable to hold the tracking data

// Get the tracking data for each frame
for (var i = 0; i < comp.duration * comp.frameRate; i++) {
    comp.time = i / comp.frameRate; // Set the time of the composition
    var pos = layer.transform.position.value; // Get the position of the layer
    var scale = layer.transform.scale.value; // Get the scale of the layer
    var rot = layer.transform.rotation.value; // Get the rotation of the layer
    var opacity = layer.transform.opacity.value; // Get the opacity of the layer

    data += pos[0] + "," + pos[1] + "," + scale[0] + "," + scale[1] + "," + rot + "," + opacity + "\n"; // Add the data to the data string
}

// Prepare the default filename based on the layer's name
var defaultFilePath = "~/Desktop/" + layer.name + ".csv";
var file = new File(defaultFilePath);

// Prompt the user to select a location for the CSV file
file = file.saveDlg("Select a location for the tracking data CSV file");
if (!file) {
    throw new Error("No location selected. Please try again.");
}

file.open("w"); // Open the file for writing
file.write(data); // Write the data to the file
file.close(); // Close the file
