function[x_var, obj_val] = svm_primal_quadprog(Xj, yj, dim_data, num_examples)
    % ///////////////////////////////////////////
    %       Solve primal of SVM with QuadProg  //
    %////////////////////////////////////////////
    Hp = diag(ones(1, dim_data+1));
    Hp(dim_data+1, dim_data+1) = 0;

    fp = zeros(dim_data+1,1);

    Ap = -[diag(yj)*Xj, yj];
    bp = -ones(num_examples, 1);
    Cp = [];
    dp = [];

    up = Inf*ones(dim_data+1, 1);
    lp = -Inf*ones(dim_data+1, 1);

    [x_var, obj_val] = quadprog(Hp,fp,Ap,bp,Cp,dp,lp,up);
end