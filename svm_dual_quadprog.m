function[alpha, obj_val] = svm_dual_quadprog(Xj, yj, num_examples)
    % ///////////////////////////////////////////
    %       Solve Dual of SVM with QuadProg    //
    %////////////////////////////////////////////
    Hd = (yj*yj').*(Xj*Xj');
    fd = -ones(num_examples,1);

    Ad = [];
    bd = [];
    Cd = yj';
    dd = 0;

    ld = zeros(num_examples, 1);
    ud = [];

    [alpha, obj_val] = quadprog(Hd,fd,Ad,bd,Cd,dd,ld,ud);
end