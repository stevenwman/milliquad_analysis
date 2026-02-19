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

files1 = ["s10leg1-1.csv", "s10leg2-2.csv", "s10leg3-3.csv"];
files2 = ["s102leg1-1.csv","s102leg2-2.csv","s102leg3-3.csv"];
files3 = ["s104leg1-1.csv","s104leg2-2.csv","s104leg3-3.csv"];


% --- Plots ---

% leg
plotSteps(files1, 1:3, RGB, styles, plot_sz, '10Hz Leg',      10, 0.3);

% 2legged
plotSteps(files2, 1:3, RGB, styles, plot_sz, '10Hz 2-Legged', 10, 0.3);

% 4legged
plotSteps(files3, 1:3, RGB, styles, plot_sz, '10Hz 4-Legged', 10, 0.3);


%% --- Data Files ---   20Hz

files1 = ["s20leg1-1.csv", "s20leg2-2.csv", "s20leg3-3.csv"];
files2 = ["s202leg1-1.csv","s202leg2-2.csv","s202leg3-3.csv"];
files3 = ["s204leg1-1.csv","s204leg2-2.csv","s204leg3-3.csv"];


% --- Plots ---

% leg
plotSteps(files1, 1:3, RGB, styles, plot_sz, '20Hz Leg',      20, 0.3);

% 2legged
plotSteps(files2, 1:3, RGB, styles, plot_sz, '20Hz 2-Legged', 20, 0.3);

% 4legged
plotSteps(files3, 1:3, RGB, styles, plot_sz, '20Hz 4-Legged', 20, 0.3);


%% --- Data Files ---   30Hz 

files1 = ["s30leg1-1.csv", "s30leg2-2.csv", "s30leg3-3.csv"];
files2 = ["s302leg1-1.csv","s302leg2-2.csv","s302leg3-3.csv"];
files3 = ["s304leg1-1.csv","s304leg2-2.csv","s304leg3-3.csv"];
files4 = ["s30w1-1.csv",   "s30w2-2.csv",   "s30w3-3.csv"];


% --- Plots ---

plotSteps(files1, 1:3, RGB, styles, plot_sz, '30Hz Leg',       30, 0.15);

plotSteps(files2, 1:3, RGB, styles, plot_sz, '30Hz 2-Legged',  30, 0.3);

plotSteps(files3, 1:3, RGB, styles, plot_sz, '30Hz 4-Legged',  30, 0.3);

plotSteps(files4, 1:3, RGB, styles, plot_sz, '30Hz wheel',     30, 0.3);

