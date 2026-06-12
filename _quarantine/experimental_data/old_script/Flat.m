%% Main Script for flat ground tracking (10, 30 & 50Hz)
clc; clear; close all; format long

% ---------- Styles ----------
RGB = orderedcolors("gem");
styles(1) =  struct('LineSpec','-','Color',RGB(1,:),'LineWidth',1);
styles(2) =  struct('LineSpec','-','Color',[0.85 0.325 0.098],'LineWidth',1);
styles(3) =  struct('LineSpec','-','Color',[0.929 0.694 0.125],'LineWidth',1);
styles(99) = struct('LineSpec','-','Color',[0 0 0],'LineWidth',1);

plot_sz = struct('Position',[0,0,800,1200]);

%% --- Data Files ---   10Hz

files1 = ["f10leg1-1.csv", "f10leg2-2.csv", "f10leg3-3.csv", "f10leg4-4.csv"];
files2 = ["f102leg1-1.csv","f102leg2-2.csv","f102leg3-3.csv","f102leg4-4.csv"];
files3 = ["f104leg1-1.csv","f104leg2-2.csv","f104leg3-3.csv","f104leg4-4.csv"];
files4 = ["f10w1-1.csv",   "f10w2-2.csv",   "f10w3-3.csv",   "f10w4-4.csv"];


% --- Plots ---

% leg
plotFlat(files1, 1:3, RGB, styles, plot_sz, '10Hz Leg', 2500, 10, 0.3);

% 2legged
plotFlat(files2, [1,2,4], RGB, styles, plot_sz, '10Hz 2-Legged', 1480, 10, 0.3);

% 4legged
plotFlat(files3, [1,2,4], RGB, styles, plot_sz, '10Hz 4-Legged', 1199, 10, 0.3);

% wheel
plotFlat(files4, 1:3, RGB, styles, plot_sz, '10Hz wheel', 910, 10, 0.3);


%% --- Data Files ---   30Hz 

files1 = ["f30leg1-1.csv", "f30leg2-2.csv", "f30leg3-3.csv", "f30leg4-4.csv"];
files2 = ["f302leg1-1.csv","f302leg2-2.csv","f302leg3-3.csv","f302leg4-4.csv"];
files3 = ["f304leg1-1.csv","f304leg2-2.csv","f304leg3-3.csv","f304leg4-4.csv"];
files4 = ["f30w1-1.csv",   "f30w2-2.csv",   "f30w3-3.csv",   "f30w4-4.csv"];


% --- Plots ---

plotFlat(files1, 1:4, RGB, styles, plot_sz, '30Hz Leg',      1100, 30, 0.15);

plotFlat(files2, 1:3, RGB, styles, plot_sz, '30Hz 2-Legged', 760,  30, 0.3);

plotFlat(files3, 1:4, RGB, styles, plot_sz, '30Hz 4-Legged', 550,  30, 0.3);

plotFlat(files4, 1:4, RGB, styles, plot_sz, '30Hz wheel',    350,  30, 0.3);


%% --- Data Files ---   50Hz 

files1 = ["f50leg1-1.csv", "f50leg2-2.csv", "f50leg3-3.csv"];
files2 = ["f502leg1-1.csv","f502leg2-2.csv","f502leg3-3.csv"];
files3 = ["f504leg1-1.csv","f504leg2-2.csv","f504leg3-3.csv"];
files4 = ["f50w1-1.csv",   "f50w2-2.csv",   "f50w3-3.csv"];


% --- Plots ---

plotFlat(files1, 1:3, RGB, styles, plot_sz, '50Hz Leg',      1960, 50, 0.35);

plotFlat(files2, 1:3, RGB, styles, plot_sz, '50Hz 2-Legged', 1280, 50, 0.35);

plotFlat(files3, 1:3, RGB, styles, plot_sz, '50Hz 4-Legged', 1060, 50, 0.35);

plotFlat(files4, 1:3, RGB, styles, plot_sz, '50Hz wheel',    620,  30, 0.25);