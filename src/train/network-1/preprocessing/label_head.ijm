//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Author: Leonardo Restivo
// date: 14.02.2020
// select frames from video and to label ROIs for further processing
// INPUT: raw video
// OUTPUT: frame (.png), binary mask (.png), coordinates file (.csv)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// the resize factor for the ROI is 64 px:
// this returns a window over the head of 128 x 128 px
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


function getMetaValue(line_number) {
	tmp = split(meta_lines[line_number],':');
	return tmp[1];
}	


// generate directories
root_dir = getDirectory("Choose ROOT Directory");
File.makeDirectory(root_dir + 'coordinates/');
File.makeDirectory(root_dir + 'frames/');
File.makeDirectory(root_dir + 'masks/');

// open movie
run("Movie (FFMPEG)...", "");

// get movie file name (for saving files and metadata)
id_movie  = File.nameWithoutExtension;
print(id_movie);

// get movie window id for selection
window_id = getImageID();

// set to extract centroid's coordinates
run("Set Measurements...", "centroid redirect=None decimal=3");

// generate the metafile
if (!File.exists(root_dir + 'meta.txt')){
	// crop frame
	setTool("rectangle");
	waitForUser("select crop area");
	getSelectionBounds(x, y, width, height);
	
	Dialog.create("Define ROI size");
	Dialog.addSlider('enlargement', 0, 24, 24);
	Dialog.addNumber("resize - width", width);
	Dialog.addNumber("resize - height", height);
	Dialog.show();
	enlargement = Dialog.getNumber();
	resize_w = Dialog.getNumber();
	resize_h = Dialog.getNumber();
		
	meta_file = File.open(root_dir + 'meta.txt');
	print(meta_file, 'x:' + x +'\n'+ 'y:'+y +'\n'+ 'width:'+ width +'\n'+'height:'+ height +'\n' +'enlargement:'+ enlargement +'\n'+'resize_width:'+ resize_w +'\n'+'resize_height:'+ resize_h +'\n');
	File.close(meta_file);
} 
else {
	meta_file = File.openAsString(root_dir +'meta.txt');
	meta_lines = split(meta_file,"\n");
	x = getMetaValue(0);
	y = getMetaValue(1);
	width = getMetaValue(2);
	height = getMetaValue(3);
	enlargement = getMetaValue(4);
	resize_w = getMetaValue(5);
	resize_h = getMetaValue(6);
}

// generate log file if it is not present in the project's directory
if (!File.exists(root_dir + 'log.txt')){
	list_of_frames = File.open(root_dir + 'log.txt');
	// print file header	
	File.append('id_movie' + '\t' + 'frame' + '\t' + 'x' + '\t' + 'y' + '\t' + 'ROI_width' + '\t' + 'ROI_height'+ '\t' + 'resize_width'+ '\t' + 'resize_height', root_dir + 'log.txt')
}

// point tool is used for 
setTool("point");
//setTool("freehand");

// by default 1000 samples are to be taken.
for (i = 0; i < 1000; i++) {

	// select movie window
	selectImage(window_id);

	// prompt user to select frame to label
	//:: the user scrolls to the frame and presses `OK`
	waitForUser('select frame to label');

	// get frame info and duplicate from the video
	Stack.getPosition(channel, slice, frame)

	makeRectangle(x, y, width, height);
	getSelectionBounds(x, y, width, height);
	
	run("Duplicate...", "title=" + slice);
	run("Size...", "width=" + resize_w + " height=" + resize_h + " depth=1 average interpolation=Bicubic");
	run("8-bit");

	// zoom in
	for (i = 0; i < 2; i++) {
		run("In [+]");	
	}

	// update the stats (i.e. frames processed) on the log window
	print("\\Update2:", "Frame #: ", slice);
	print("\\Update3:", "total # of frames: ", i+1);

	// prompt user to click on the ROI !!!! THE ROI ENLARGMENT IS HARD CODED !!!!
	// ask users to define the enlargment factor!
	waitForUser('click on ROI');
	run("Enlarge...", "enlarge=" + enlargement);

	// shortcut for saving metadata in the filename
	fname_to_save = id_movie + '_' + slice + '_' + resize_w + 'x' + resize_h;
	
	// save frame
	saveAs("PNG", root_dir + 'frames/' + fname_to_save + ".png");

	// log metadata on the LOG file
	File.append(id_movie + '\t' + slice + '\t' + x + '\t' + y + '\t' + width + '\t' + height + '\t' + resize_w + '\t' + resize_h, root_dir + 'log.txt')
	
	// save COORDINATES
	run("Measure");		
	saveAs("Results", root_dir +'coordinates/' + fname_to_save + ".txt");
	
	// final cleanup before loading next image
	run("Clear Results");
	
	// body is set to 255, background to 0
	setForegroundColor(255, 255, 255);
	run("Fill", "slice");
	run("Make Inverse");
	setBackgroundColor(0, 0, 0);
	run("Clear", "slice");
	
	// save to MASK folder
	saveAs("PNG", root_dir +'masks/' + fname_to_save + ".png");

	close();		
}
