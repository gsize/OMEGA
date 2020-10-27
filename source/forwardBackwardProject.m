classdef forwardBackwardProject
    %FORWARDBACKWARDPROJECT A MATLAB class for computing the backward and
    %forward projections in OMEGA or forming the entire system matrix
    %   Simplified version of the forward_backward_projections_example.m
    %   (an example of this class is included in the same file).
    
    properties
        OProperties
        n_meas
        nn
        gaussK
        sens
        osa_iter
        index
        sysmat
        trans
        TOF
    end
    
    methods
        function obj = forwardBackwardProject(options)
            %FORWARDBACKWARDPROJECT Construct an instance of this class
            %   Input is the options struct generated by any of the
            %   main-files. All parameters should be adjusted in the
            %   corresponding main-file, though manual adjustment is also
            %   possible. However, the object should be always re-created
            %   after any parameters are changed, otherwise the
            %   reconstructions may not work or might produce wrong
            %   results.
            obj.OProperties = options;
            [obj.index, obj.n_meas, obj.OProperties.subsets] = index_maker(obj.OProperties.Nx, obj.OProperties.Ny, obj.OProperties.Nz, obj.OProperties.subsets, obj.OProperties.use_raw_data, ...
                obj.OProperties.machine_name, obj.OProperties, obj.OProperties.Nang, obj.OProperties.Ndist, obj.OProperties.TotSinos, obj.OProperties.NSinos);
            
            obj.nn = [int64(0);int64(cumsum(obj.n_meas))];
            if iscell(obj.index)
                obj.index = cell2mat(obj.index);
            end
            if obj.OProperties.subsets > 1
                obj.n_meas = [int64(0);int64(cumsum(obj.n_meas))];
            end
            if obj.OProperties.subsets == 1
                obj.osa_iter = 1;
            end
            
            
            if ~isfield(obj.OProperties,'use_machine')
                obj.OProperties.use_machine = 0;
            end
            if ~isfield(obj.OProperties,'TOF_bins') || obj.OProperties.TOF_bins == 0
                obj.OProperties.TOF_bins = 1;
            end
            if ~isfield(obj.OProperties,'TOF_width') || obj.OProperties.TOF_bins == 0
                obj.OProperties.TOF_width = 0;
            end
            
            obj.TOF = obj.OProperties.TOF_bins > 1;
            
            [obj.gaussK, obj.OProperties] = PSFKernel(obj.OProperties);
            if obj.OProperties.implementation == 1 || obj.OProperties.implementation == 4
                obj.sens = zeros(obj.OProperties.Nx*obj.OProperties.Ny*obj.OProperties.Nz,obj.OProperties.subsets);
            else
                obj.sens = zeros(obj.OProperties.Nx*obj.OProperties.Ny*obj.OProperties.Nz,obj.OProperties.subsets,'single');
            end
            
            
            pseudot = uint32(obj.OProperties.pseudot);
            temp = pseudot;
            if ~isempty(temp) && temp > 0
                for kk = int32(1) : temp
                    pseudot(kk) = int32(obj.OProperties.cryst_per_block + 1) * kk;
                end
            elseif temp == 0
                pseudot = [];
            end
            % Number of rings
            R = double(obj.OProperties.diameter);
            FOVax = obj.OProperties.FOVa_x;
            FOVay = obj.OProperties.FOVa_y;
            FOVax = double(FOVax);
            FOVay = double(FOVay);
            axial_fow = double(obj.OProperties.axial_fov);
            rings = options.rings;
            blocks = uint32(rings + length(pseudot) - 1);
            block1 = uint32(0);
            
            [x, y, z_det, obj.OProperties] = get_coordinates(obj.OProperties, blocks, pseudot);
            obj.OProperties.x = x;
            obj.OProperties.y = y;
            obj.OProperties.z_det = z_det;
            
            [normalization_correction, randoms_correction, obj.OProperties] = set_up_corrections(obj.OProperties, blocks);
            obj.OProperties.normalization_correction = normalization_correction;
            obj.OProperties.randoms_correction = randoms_correction;
            
            if obj.OProperties.use_raw_data
                size_x = uint32(obj.OProperties.det_w_pseudo);
            else
                size_x = uint32(obj.OProperties.Nang*obj.OProperties.Ndist);
            end
            
            [obj.OProperties, obj.OProperties.lor_a, obj.OProperties.xy_index, obj.OProperties.z_index, obj.OProperties.LL, obj.OProperties.summa, obj.n_meas,~,~,discard] = ...
                form_subset_indices(obj.OProperties, obj.n_meas, obj.OProperties.subsets, obj.index, size_x, y, z_det, blocks, false, obj.TOF);
            if ~obj.OProperties.precompute_lor
                obj.OProperties.lor_a = uint16(0);
            end
            if obj.OProperties.subsets == 1
                obj.nn = int64(obj.n_meas);
                if obj.OProperties.precompute_lor
                    obj.index = find(discard);
                end
            end
            
            etaisyys_x = (R-FOVax)/2;
            etaisyys_y = (R-FOVay)/2;
            if obj.OProperties.implementation == 2 || obj.OProperties.implementation == 3 || obj.OProperties.implementation == 5
                zz = linspace(single(0), single(axial_fow), obj.OProperties.Nz + 1);
                xx = single(linspace(etaisyys_x, R - etaisyys_x, obj.OProperties.Nx + 1));
                yy = single(linspace(etaisyys_y, R - etaisyys_y, obj.OProperties.Ny + 1));
            else
                zz = linspace(double(0), double(axial_fow), obj.OProperties.Nz + 1);
                xx = double(linspace(etaisyys_x, R - etaisyys_x, obj.OProperties.Nx + 1));
                yy = double(linspace(etaisyys_y, R - etaisyys_y, obj.OProperties.Ny + 1));
            end
            zz=zz(2*block1+1:2*blocks);
            
            % Distance of adjacent pixels
            dx = diff(xx(1:2));
            dy = diff(yy(1:2));
            dz = diff(zz(1:2));
            
            if obj.OProperties.projector_type == 2 || obj.OProperties.projector_type == 3
                obj.OProperties.x_center = xx(1 : end - 1)' + dx/2;
                obj.OProperties.y_center = yy(1 : end - 1)' + dy/2;
                obj.OProperties.z_center = zz(1 : end - 1)' + dz/2;
                temppi = min([obj.OProperties.FOVa_x / obj.OProperties.Nx, obj.OProperties.axial_fov / obj.OProperties.Nz]);
                if obj.OProperties.tube_width_z > 0
                    temppi = max([1,round(obj.OProperties.tube_width_z / temppi)]);
                else
                    temppi = max([1,round(obj.OProperties.tube_width_xy / temppi)]);
                end
                temppi = temppi * temppi * 4;
                if obj.OProperties.apply_acceleration
                    if obj.OProperties.tube_width_z == 0
                        dec = uint32(sqrt(obj.OProperties.Nx^2 + obj.OProperties.Ny^2) * temppi);
                    else
                        dec = uint32(sqrt(obj.OProperties.Nx^2 + obj.OProperties.Ny^2 + obj.OProperties.Nz^2) * temppi);
                    end
                else
                    dec = uint32(0);
                end
                obj.OProperties.dec = dec;
            elseif (options.projector_type == 1 && obj.TOF)
                obj.OProperties.x_center = xx(1);
                obj.OProperties.y_center = yy(1);
                obj.OProperties.z_center = zz(1);
                if obj.OProperties.apply_acceleration && obj.OProperties.n_rays_transaxial * obj.OProperties.n_rays_axial == 1
                    dec = uint32(sqrt(obj.OProperties.Nx^2 + obj.OProperties.Ny^2 + obj.OProperties.Nz^2) * 2);
                else
                    dec = uint32(0);
                end
                obj.OProperties.dec = dec;
            else
                obj.OProperties.x_center = xx(1);
                obj.OProperties.y_center = yy(1);
                obj.OProperties.z_center = zz(1);
                obj.OProperties.dec = uint32(0);
            end
            
            if obj.OProperties.projector_type == 3
                voxel_radius = (sqrt(2) * obj.OProperties.voxel_radius * dx) / 2;
                bmax = obj.OProperties.tube_radius + voxel_radius;
                b = linspace(0, bmax, 10000)';
                b(obj.OProperties.tube_radius > (b + voxel_radius)) = [];
                b = unique(round(b*10^3)/10^3);
                V = volumeIntersection(obj.OProperties.tube_radius, voxel_radius, b);
                Vmax = (4*pi)/3*voxel_radius^3;
                bmin = min(b);
            else
                V = 0;
                Vmax = 0;
                bmin = 0;
                bmax = 0;
            end
            if obj.OProperties.implementation == 2 || obj.OProperties.implementation == 3
                obj.OProperties.V = single(V);
                obj.OProperties.Vmax = single(Vmax);
                obj.OProperties.bmin = single(bmin);
                obj.OProperties.bmax = single(bmax);
            else
                obj.OProperties.V = double(V);
                obj.OProperties.Vmax = double(Vmax);
                obj.OProperties.bmin = double(bmin);
                obj.OProperties.bmax = double(bmax);
            end
            % Multi-ray Siddon
            if obj.OProperties.implementation > 1 && obj.OProperties.n_rays_transaxial > 1 && ~obj.OProperties.precompute_lor && obj.OProperties.projector_type == 1
                [x,y] = getMultirayCoordinates(obj.OProperties);
                obj.OProperties.x = x;
                obj.OProperties.y = y;
            end
            obj.trans = false;
        end
        
        function y = forwardProject(obj, input, varargin)
            %FORWARDPROJECT Computes the forward projection between the
            %object and the input vector.
            %   Output is stored in the y-vector. PSF blurring is performed
            %   if it has been selected.
            % Inputs:
            %   obj = The forwardBackwardProject object created by
            %   forwardBackwardProject
            %   input = The input vector, i.e. this computes A * input
            %   subset_number = Current subset, used to select the correct
            %   LORs. Applicable only when using subsets, omit otherwise.
            % Outputs:
            %   y = The result of y = A * input
            
            if nargin >=3 && ~isempty(varargin{1})
                obj.osa_iter = varargin{1};
            elseif obj.OProperties.subsets > 1 && nargin == 2
                error('When using subsets you must specify the current subset number (e.g. y = forwardProject(A, input, subset_number))')
            end
            
            if obj.OProperties.use_psf
                input = computeConvolution(input, obj.OProperties, obj.OProperties.Nx, obj.OProperties.Ny, obj.OProperties.Nz, obj.gaussK);
            end
            y = forward_project(obj.OProperties, obj.index(obj.nn(obj.osa_iter) + 1:obj.nn(obj.osa_iter+1)), obj.nn(obj.osa_iter + 1) - obj.nn(obj.osa_iter), input, [obj.nn(obj.osa_iter) + 1 , obj.nn(obj.osa_iter+1)], ...
                obj.osa_iter, true);
        end
        
        
        function [f, varargout] = backwardProject(obj, input, varargin)
            %BACKWARDPROJECT Computes the backprojection between the object
            %and the input vector. Can also (optionally) compute the
            %sensitivity image.
            %   Output is stored in the f-vector. PSF blurring is performed
            %   if it has been selected.
            % Inputs:
            %   obj = The forwardBackwardProject object created by
            %   forwardBackwardProject
            %   input = The input vector, i.e. this computes A' * input
            %   subset_number = Current subset, used to select the correct
            %   LORs. Applicable only when using subsets, omit otherwise.
            % Outputs:
            %   f = The result of f = A' * input
            %   obj = The modified object. Sensitivity image is stored in
            %   sens. Can be omitted if sensitivity image is not required.
            
            if nargin >=3 && ~isempty(varargin{1})
                obj.osa_iter = varargin{1};
            elseif obj.OProperties.subsets > 1 && nargin == 2
                error('When using subsets you must specify the current subset number (e.g. f = backwardProject(A, input, subset_number))')
            end
            if nargout >= 2
                iter = 1;
            else
                iter = 10;
            end
            if iter == 1
                [f, norm] = backproject(obj.OProperties, obj.index(obj.nn(obj.osa_iter) + 1:obj.nn(obj.osa_iter+1)), obj.nn(obj.osa_iter + 1) - obj.nn(obj.osa_iter), input, ...
                    [obj.nn(obj.osa_iter) + 1,obj.nn(obj.osa_iter+1)], obj.osa_iter, true);
                
                obj.sens(:,obj.osa_iter) = norm;
                if obj.OProperties.use_psf
                    obj.sens(:,obj.osa_iter) = computeConvolution(obj.sens(:,obj.osa_iter), obj.OProperties, obj.OProperties.Nx, obj.OProperties.Ny, obj.OProperties.Nz, obj.gaussK);
                end
            else
                f = backproject(obj.OProperties, obj.index(obj.nn(obj.osa_iter) + 1:obj.nn(obj.osa_iter+1)), obj.nn(obj.osa_iter + 1) - obj.nn(obj.osa_iter), input, ...
                    [obj.nn(obj.osa_iter) + 1,obj.nn(obj.osa_iter+1)], obj.osa_iter, true);
            end
            if obj.OProperties.use_psf
                f = computeConvolution(f, obj.OProperties, obj.OProperties.Nx, obj.OProperties.Ny, obj.OProperties.Nz, obj.gaussK);
            end
            if nargout >= 2
                varargout{1} = obj;
            end
        end
        function f = mtimes(obj, input)
            %MTIMES Automatically compute either the forward projection or
            %backprojection, based on the input vector length.
            %   Backprojection is selected if transpose operator is used.
            if obj.trans == true || size(input,1) == obj.n_meas(end) || size(input,2) == obj.n_meas(end) || size(input,1) == obj.n_meas(end) * obj.OProperties.TOF_bins || size(input,2) == obj.n_meas(end) * obj.OProperties.TOF_bins
                if size(input,2) == obj.n_meas(end)
                    input = input';
                end
                f = backwardProject(obj, input);
            elseif size(input,1) == obj.OProperties.Nx*obj.OProperties.Ny*obj.OProperties.Nz || size(input,2) == obj.OProperties.Nx*obj.OProperties.Ny*obj.OProperties.Nz  || obj.trans == false
                if size(input,2) == obj.OProperties.Nx*obj.OProperties.Ny*obj.OProperties.Nz
                    input = input';
                end
                f = forwardProject(obj, input);
            end
        end
        function A = formMatrix(obj, varargin)
            %FORMMATRIX Forms the PET system matrix for the current subset
            %or, if no subsets are used, the entire matrix
            %   Corrections are applied to the matrix if they were
            %   selected. Always uses implementation 1 regardless of
            %   choice.
            % Inputs:
            %   obj = The forwardBackwardProject object created by
            %   forwardBackwardProject
            %   subset_number = Current subset, used to select the correct
            %   LORs. Applicable only when using subsets, omit otherwise.
            % Output:
            %   A = The system matrix for the current subset or the entire
            %   system matrix if subsets are set to 1 or omitted.
            if obj.OProperties.implementation ~= 1
                warning('Implementation not set to 1, reverting to 1 to compute the system matrix')
                obj.OProperties.implementation = 1;
            end
            if nargin >= 2 && ~isempty(varargin{1})
                obj.osa_iter = varargin{1};
            elseif obj.OProperties.subsets > 1 && nargin == 1
                error('When using subsets you must specify the current subset number (e.g. sysMat = formMatrix(A, subset_number))')
            end
            A = forward_project(obj.OProperties, obj.index(obj.nn(obj.osa_iter) + 1:obj.nn(obj.osa_iter+1)), obj.nn(obj.osa_iter + 1) - obj.nn(obj.osa_iter), [], [obj.nn(obj.osa_iter) + 1 , obj.nn(obj.osa_iter+1)], ...
                obj.osa_iter, true, true);
        end
        function obj = transpose(obj)
            obj.trans = true;
        end
    end
end

