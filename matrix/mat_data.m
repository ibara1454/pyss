% Generate eigenvalues and eigenvectors for test datas
% by using matlab's eig function
%
% Note that each data of matrices A and B are sparse

prec = 16;

C = {'bcsstk11.mtx',   'mhd4800a.mtx';
     'bcsstm11.mtx',   'mhd4800b.mtx';
     'bcsst11eig.mtx', 'mhd4800eig.mtx';};
 
for data = C
    in_A_path = data{1};
    in_B_path = data{2};
    out_eig_path = data{3};
    if exist(out_eig_path, 'file')
        continue;
    end
    [A, rows, cols, entries] = mmread(in_A_path);
    [B, rows, cols, entries] = mmread(in_B_path);
    [V, D] = eig(full(A), full(B));

    mmwrite(out_eig_path, diag(D), 'precision', prec)
end
% if not(exist(out_vec_path, 'file'))
%     dlmwrite(out_vec_path, V, 'precision', prec)
% end
