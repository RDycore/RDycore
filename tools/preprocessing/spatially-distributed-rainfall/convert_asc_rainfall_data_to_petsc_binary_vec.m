function convert_asc_rainfall_data_to_petsc_binary_vec(petsc_dir)
% convert_asc_rainfall_data_to_petsc_binary_vec
%
% Converts rainfall data in the ASCII raster format to a PETSc binary format 
% that is read by RDycore as a spatially distributed rainfall dataset. The
% binary data is written in PETSc's int32 and int64 format.
%
% Example usage:
% convert_asc_rainfall_data_to_petsc_binary_vec('/Users/bish218/projects/petsc/petsc_v3.21.4/')

inp_dir = 'daymet';

addpath([petsc_dir 'share/petsc/matlab/']); 

inp_asc_dir = [inp_dir '/asc'];
out_bin_dir = [inp_dir '/bin'];
system(['mkdir -p ' out_bin_dir]);

files = dir([inp_asc_dir '/*.asc']);

for ii = 1:length(files)

    fname = [inp_asc_dir '/' files(ii).name];

    % get date information from the filename
    mo = files(ii).name( 1: 2);
    dd = files(ii).name( 3: 4);
    yy = files(ii).name( 5: 8);
    hh = files(ii).name( 9:10);
    mm = files(ii).name(11:12);

    % open the file
    fid = fopen(fname,'r');

    ncols        = -1;
    nrows        = -1;
    xlc          = -1;
    ylc          = -1;
    cellsize     = -1;
    nodata_value = -1;

    % read 6 lines of header information
    for iline = 1:6
        line = fgetl(fid);
        tmp = strsplit(line, '\t');

        switch lower(tmp{1})
            case 'ncols'
                ncols = str2num(tmp{2});
            case 'nrows'
                nrows = str2num(tmp{2});
            case 'xllcorner'
                xlc = str2num(tmp{2});
            case 'yllcorner'
                ylc = str2num(tmp{2});
            case 'cellsize'
                cellsize  = str2num(tmp{2});
            case 'nodata_value'
                nodata_value  = str2num(tmp{2});
            otherwise
                error(['Unknown value: ' tmp{1}])
        end

    end

    % read the data
    data = fscanf(fid,'%f');

    if (length(data) ~= nrows*ncols)
        error('Size of data (=%d) read does not match the product of number of rows (=%d) and cols (=%d)',length(data),nrows,ncols);
    end

    % Replace nodata with zeros
    data(data == nodata_value) = 0.0;

    vec = [ncols nrows xlc ylc cellsize data']';
    
    % write out the data

    for kk = 1:2
        switch kk
            case 1
                indices = 'int32';
            case 2
                indices = 'int64';
        end
        out_fname = [out_bin_dir '/' yy '-' mo '-' dd ':' hh '-' mm '.' indices '.bin'];

        disp(out_fname);
        PetscBinaryWrite(out_fname,vec,'indices',indices);
    end
end


