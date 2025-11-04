% From Donghui Xu
% https://github.com/donghuix/Setup-E3SM-Mac/blob/master/matlab-scripts-to-process-inputs/generate_dlnd.m
function generate_dlnd(QDRAI,QOVER,lat,lon,time,startdate,isleap,fname_out)

% startdate should be yyyy-mm-dd format
    
    QRUNOFF = QOVER + QDRAI; % total runoff is the sum of surface and subsurface runoff

    if length(size(QDRAI)) == 2 
        isgrid2d = false;
        [nlon,nt] = size(QDRAI);
        nlat = 1;
        disp(['DLND forcing is 1D with dimensions lat = 1, lon = ' num2str(nlon)]);
    elseif length(size(QDRAI)) == 3 
        isgrid2d = true;
        [nlon,nlat,nt] = size(QDRAI);
        disp(['DLND forcing is 2D with dimensions lat = ' num2str(nlat) ', lon = ' num2str(nlon)]);
    else
        error('check dimension of QDRAI!!!');
    end
    disp(['TIME dimension is nt = ' num2str(nt)]);
    
%     if isgrid2d
%         assert(nlon == length(lon) && nlat == length(lat) && nt == length(time));
%     else
%         assert(nlon == length(lon) && nlon == length(lat) && nt == length(time));
%     end
    
    netcdf.setDefaultFormat('NC_FORMAT_64BIT'); % To write for large file
    ncid_out = netcdf.create(fname_out,'NC_CLOBBER');
    
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%
%                           Define dimensions
%
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dimid(1) = netcdf.defDim(ncid_out,'lat',nlat);
    dimid(2) = netcdf.defDim(ncid_out,'lon',nlon);
    dimid(3) = netcdf.defDim(ncid_out,'time',nt);
    
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%
%                           Define variables
%
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    varnames = {'QDRAI', 'QOVER', 'QRUNOFF', 'lat', 'lon', 'time'};
    nvars = length(varnames);
    ivar = 1;
    varid(ivar) = netcdf.defVar(ncid_out,'QDRAI',6,[dimid(2),dimid(1),dimid(3)]); 
    netcdf.putAtt(ncid_out,ivar-1,'standard_name','subsurface runoff');
    netcdf.putAtt(ncid_out,ivar-1,'units','mm/s');
    
    ivar = 2;
    varid(ivar) = netcdf.defVar(ncid_out,'QOVER',6,[dimid(2),dimid(1),dimid(3)]); 
    netcdf.putAtt(ncid_out,ivar-1,'standard_name','surface runoff');
    netcdf.putAtt(ncid_out,ivar-1,'units','mm/s');
    
    ivar = 3;
    varid(ivar) = netcdf.defVar(ncid_out,'QRUNOFF',6,[dimid(2),dimid(1),dimid(3)]); 
    netcdf.putAtt(ncid_out,ivar-1,'standard_name','total runoff');
    netcdf.putAtt(ncid_out,ivar-1,'units','mm/s');
    
    ivar = 4;
    %if isgrid2d
    %    varid(ivar) = netcdf.defVar(ncid_out,'lat',6,dimid(1)); 
    %else
    %    varid(ivar) = netcdf.defVar(ncid_out,'lat',6,[dimid(2) dimid(1)]); 
    %end
    varid(ivar) = netcdf.defVar(ncid_out,'LATIXY',6,[dimid(2) dimid(1)]); 
    netcdf.putAtt(ncid_out,ivar-1,'standard_name','latitude');
    netcdf.putAtt(ncid_out,ivar-1,'long_name','latitude');
    netcdf.putAtt(ncid_out,ivar-1,'units','degrees_north');
    %netcdf.putAtt(ncid_out,ivar-1,'units','radians');
    netcdf.putAtt(ncid_out,ivar-1,'axis','Y');
    
    ivar = 5;
    %if isgrid2d
    %    varid(ivar) = netcdf.defVar(ncid_out,'lon',6,dimid(2)); 
    %else
    %    varid(ivar) = netcdf.defVar(ncid_out,'lon',6,[dimid(2) dimid(1)]); 
    %end
    varid(ivar) = netcdf.defVar(ncid_out,'LONGXY',6,[dimid(2) dimid(1)]); 
    netcdf.putAtt(ncid_out,ivar-1,'standard_name','longitude');
    netcdf.putAtt(ncid_out,ivar-1,'long_name','longitude');
    netcdf.putAtt(ncid_out,ivar-1,'units','degrees_east');
    %netcdf.putAtt(ncid_out,ivar-1,'units','radians');
    netcdf.putAtt(ncid_out,ivar-1,'axis','X');
    
    ivar = 6;
    varid(ivar) = netcdf.defVar(ncid_out,'time',6,dimid(3));
    netcdf.putAtt(ncid_out,ivar-1,'standard_name','time');
    if isleap
        netcdf.putAtt(ncid_out,ivar-1,'calendar','gregorian');
    else
        netcdf.putAtt(ncid_out,ivar-1,'calendar','noleap');
    end
    netcdf.putAtt(ncid_out,ivar-1,'units',['days since ' startdate ' 00:00:00']);
    netcdf.putAtt(ncid_out,ivar-1,'axis','T');
    
    varidg = netcdf.getConstant('GLOBAL');

    [~,user_name]=system('echo $USER');
    netcdf.putAtt(ncid_out,varidg,'Created_by' ,user_name(1:end-1));
    netcdf.putAtt(ncid_out,varidg,'Created_on' ,datestr(now,'ddd mmm dd HH:MM:SS yyyy '));
    netcdf.endDef(ncid_out);
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%
%                           Copy variables
%
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if ~isgrid2d
        QDRAI = reshape(QDRAI,[nlon,nlat,nt]);
        QOVER = reshape(QOVER,[nlon,nlat,nt]);
        QRUNOFF = QOVER + QDRAI; % 
    end

    ivar = 1;
    netcdf.putVar(ncid_out,ivar-1,QDRAI);
    ivar = 2;
    netcdf.putVar(ncid_out,ivar-1,QOVER);
    ivar = 3;
    netcdf.putVar(ncid_out,ivar-1,QRUNOFF);
    ivar = 4;
    netcdf.putVar(ncid_out,ivar-1,lat);
    ivar = 5;
    netcdf.putVar(ncid_out,ivar-1,lon);
    ivar = 6;
    netcdf.putVar(ncid_out,ivar-1,time);
    
    netcdf.close(ncid_out);
    
end