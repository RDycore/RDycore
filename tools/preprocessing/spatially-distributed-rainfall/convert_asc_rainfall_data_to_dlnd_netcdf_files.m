function convert_asc_rainfall_data_to_dlnd_netcdf_files()
% convert_asc_rainfall_data_to_dlnd_netcdf_files
%
% Converts rainfall data in the ASCII raster format to netcdf file that can 
% read by DLND (data land) component of E3SM.
%
% Example usage:
% convert_asc_rainfall_data_to_dlnd_netcdf_files()

rainfall_data_name = 'daymet';
inp_dir = rainfall_data_name;

inp_asc_dir  = [inp_dir '/asc'];
out_dlnd_dir = [inp_dir '/dlnd'];

system(['mkdir -p ' out_dlnd_dir]);


files = dir([inp_asc_dir '/*.asc']);

nfiles = length(files);

mo = zeros(nfiles,1);
dd = zeros(nfiles,1);
yy = zeros(nfiles,1);
hh = zeros(nfiles,1);
mm = zeros(nfiles,1);


for ii = 1:nfiles

    fname = [inp_asc_dir '/' files(ii).name];
    disp(fname)

    % get date information from the filename
    mo(ii) = str2num(files(ii).name( 1: 2));
    dd(ii) = str2num(files(ii).name( 3: 4));
    yy(ii) = str2num(files(ii).name( 5: 8));
    hh(ii) = str2num(files(ii).name( 9:10));
    mm(ii) = str2num(files(ii).name(11:12));

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
    fclose(fid);

    if (length(data) ~= nrows*ncols)
        error('Size of data (=%d) read does not match the product of number of rows (=%d) and cols (=%d)',length(data),nrows,ncols);
    end

    % Replace nodata with zeros
    data(data == nodata_value) = 0.0;

    unit_conversion = 1/3600; % mm/hr --> mm/s
    data = data*unit_conversion;

    if (ii == 1)
        %data_2d = zeros(nrows,ncols,nfiles);
        data_2d = zeros(ncols,nrows,nfiles);
    end

    %data_2d(:,:,ii) = flipud(reshape(data, ncols, nrows)');
    data_2d(:,:,ii) = flipud(reshape(data, ncols, nrows)')';

end

% After checking with Donghui, an offset of cellsize/2 was not added in the 
% X/Y values
X = xlc : cellsize : xlc + (ncols-1)*cellsize;
Y = ylc : cellsize : ylc + (nrows-1)*cellsize;
[X,Y] = meshgrid(X,Y);

proj = projcrs(32610);
[lat,lon] = projinv(proj,X,Y);

% Change [-180,180] to [0,360]
lon = 360+lon;

% % Change deg to rad
% lat = lat * pi/180;
% lon = lon * pi/180;

startdate = sprintf('%04d-%02d-%02d',yy(1),mo(1),dd(1));

QDRAI  = data_2d*0;
QOVER  = data_2d;
time   = dd - dd(1) + hh/24;
time = time - time(1);
isleap = false;
fname_out = [out_dlnd_dir '/' rainfall_data_name '_' sprintf('%02d-%02d-%04d_to_%02d-%02d-%04d',mo(1),dd(1),yy(1),mo(end),dd(end),yy(end)) '.nc'];

disp(fname_out)
generate_dlnd(QDRAI,QOVER,lat',lon',time,startdate,isleap,fname_out)
