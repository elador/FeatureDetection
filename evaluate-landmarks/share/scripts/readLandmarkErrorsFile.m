function [diff, filenames] = readLandmarkErrorsFile(filename)

fid = fopen(filename);

line = fgets(fid);
diff = [];
filenames = [];
filename = '';
while ischar(line)
    if (line(1) == '#')
        filename = line;
    else
        val = sscanf(line, '%f %s %s');
        diff = [diff; val(1)];
        filenames = [filenames; filename]; % note: breaks if filenames have different length
    end
    line = fgets(fid);
end

fclose(fid);

end