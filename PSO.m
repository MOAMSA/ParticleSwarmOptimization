function PSO(obj_fun,num_par,low_bo,up_bo,pop_size,max_itr,cog_para,soc_para,cons_fac,stall_gen_lim,fun_tol) 
%
%%%%% This function optimizes a function using Particle Swarm Optimization
%%%%% (PSO) methods wich is an evolutionary method.
%
%%%%% Defining Objective Function (OF).
%
OF = obj_fun ;
%
%%%%% Initializing method variables.
%
%%%% Size of the swarm.
%
if isempty(pop_size) == 1
    PS = 20 ;
else
    PS = pop_size ;
end
%
%%%% Dimension of the problem.
%
NP = num_par ;
%
%%%% Lower bound of function.
%
if isempty(low_bo) == 1
    LB = -10;
else
    LB = low_bo ;
end
%
%%%% Upper bound of function.
%
if isempty(up_bo) == 1
    UB = 10;
else
    UB = up_bo ;
end
%
%%%% Maximum number of iterations.
%
if  isempty(max_itr) == 1
    MI = 200 ;
else
    MI = max_itr ;
end
%
%%%% Maximum number of iterations.
%
if  isempty(stall_gen_lim) == 1
    SGL = 50 ;
else
    SGL = stall_gen_lim ;
end
%
%%%% Maximum number of iterations.
%
if  isempty(fun_tol) == 1
    FUNTOL = 1e-6 ;
else
    FUNTOL = fun_tol ;
end
%
%%%% Cognitive parameter.
% 
if isempty(cog_para ) == 1
    C1 = 2 ;
else C1 = cog_para ;
end
%
%%%% Social parameter.
%
if isempty(soc_para) == 1
    C2 = 4-C1 ;
else C2 = soc_para ;
end
%
%%%% Constriction factor.
%
if  isempty(cons_fac) == 1
    CF = 1 ;
else CF = cons_fac ;
end
%
%%%%% Initializing swarm and velocities.
%
%%%% Random population of  continuous values.
%
PAR = zeros(PS,NP) ;
for ii = 1:NP
    PAR(:,ii)= (UB(1,ii)-LB(1,ii))*rand(PS,1)+LB(1,ii) ;
end
%
%%%% Random velocities.
%
V = rand(PS,NP) ; 
%
%%%%% Evaluate initial population.
%
%%%% Calculates population cost using OF.
%
COST = zeros (PS,1) ;
for ii = 1:PS
    COST(ii,1) = feval(OF,PAR(ii,:)) ;
end
% %
% %%%% Recording minimum cost.
% %
 MC(1) = min(COST) ; 
%
%%%% Recording mean of cost.
%
MEC(1) = mean(COST) ; 
%
%%%% Initializing global minimum cost.
%
[GMC(1) IND] = min(COST) ;
GMCP(1,:) = PAR(IND,:) ;
%
%%%% Initializing local minimum for each particle.
%
%%% Location of local minima.
%
LP = PAR ; 
%
%%% Cost of local mimima.
%
LC = COST ;
%
%%%% Finding best particle in initial population.
%
[GC IND] = min(GMC) ;
GP = GMCP(IND,:) ;
GCR(1,:) = GC ;
%
%%%%% Start iterations.
%
%%%% Counting iterations.
%
ITR = 0 ;
%
%%%% Checking iteration exceedance. 
%
while ITR < MI
    ITR = ITR + 1 ;
    %
    %%%% Updating velocity.
    %
    %%% Inertia weigth.
    %
    W = (MI-ITR)/MI ; 
    %%% Random numbers.
    %
    R1 = rand(PS,NP) ;
    R2 = rand(PS,NP) ;
    %
    %%% Calculating new velocity.
    %
    V = CF*(W*V+C1*R1.*(LP-PAR)+C2*R2.*(ones(PS,1)*GMCP(ITR,:)-PAR)) ;
    %
    PAR = PAR+V ;
    % 
    %%%% Setting over and under limits.
    %
    UBM = zeros(PS,NP) ;
    LBM = zeros(PS,NP) ;
    for ii =1:NP
        UBM(:,ii) = UB(ii)*ones(PS,1) ;
        LBM(:,ii) = LB(ii)*ones(PS,1) ;
    end
    OL = PAR<=UBM ;
    UL = PAR>=LBM ;
    PAR = PAR.*OL+not(OL).*UBM ;
    PAR = PAR.*UL+not(UL).*LBM ;
    %
    %%%%% Evaluating the new swarm.
    %
    %%%% Evaluating the cost of swarms.
    %
    for ii = 1:PS
        COST(ii,1) = feval(OF,PAR(ii,:)) ;
    end
    %
    %%%%% Updating the best local position for each particle.
    %
    %%% Location of local minima.
    %
    for ii = 1:NP
        if COST(ii,1) < LC(ii,1)
            LP(ii,1) = PAR(ii,1) ;
        end
    end
    %
    %%% Cost of local mimima.
    %
    LC = COST ;
    %
    %%%% Updating particle positions.
    %
    [GMC(ITR+1) IND] = min(COST) ;
    GMCP(ITR+1,:) = PAR(IND,:) ;
    [GC IND] = min(GMC) ;
    GP = GMCP(IND,:) ;
    GCR(ITR+1) = GC ; 
    %
    %%%%% Printing output of each iteration.
    %
    fprintf('< ITERATION NO. = %d >  < GLOBAL COST = %f >\n',ITR,GC) ;
    %
    %%%% Updating identifiers.
    %
    MC(ITR+1) = min(COST) ; 
    MEC(ITR+1) = mean(COST) ;
    %
    %%%%% Checking stall generation limit.
    %
    if ITR > SGL
        BFE = GMC((ITR-SGL):end) ;
        FCH = 0 ;
        for jj = 1:SGL
             FCH = FCH + 0.5^(SGL-jj)*(abs(BFE(jj+1)-BFE(jj))/(abs(BFE(jj))+1)) ;
        end
        FCH = FCH/SGL ;
    else
        FCH = inf ;
    end
    %
    %%%%% Checking last improvement.
    %
     if (ITR > 1) && isfinite(GMC(ITR))
            if GCR(ITR-1) > GCR(ITR)
                LIMP = ITR ;
            end
     else
         LIMP = 1 ;
     end
    %
    %%%%% Checking termination reason.
    %
    if FCH <= FUNTOL
        fprintf('Optimization terminated: Average change in the fitness value less than function tolerance.' ) ;
        break ;
    elseif (ITR-LIMP) >SGL
         fprintf('Optimization terminated: Stall generation limit exceeded.' ) ;
         break ;
    elseif ITR >= MI
        fprintf('Optimization terminated: Maximum number of generations exceeded.' ) ;
    end       
end
%
%%%%% Plotting outputs.
%
ITRS = 0:length(MC)-1 ;
plot(ITRS,MEC,'r',ITRS,GC,'b') ;
xlabel('Generations Number') ;
ylabel('Cost') ;
legend('Mean cost','Best cost') ;
%
%%%%% Printing the results.
%
%%%%% Printing output of each iteration.
%
GLOBAL_PARAMETER = GP
GLOBAL_COST = GC
%
%%%%% End of function.
%
end