function run_spm_dcm(manifest_file)
%RUN_SPM_DCM Convert cleaned epochs and invert prespecified SPM12 DCMs.
%
% Called by run_dcm_pilot.py. This is actual SPM DCM inversion, not a
% connectivity approximation. Picture data use DCM-ERP; rest uses DCM-CSD.

spm('defaults', 'EEG');
spm_jobman('initcfg');

manifest = jsondecode(fileread(manifest_file));
output_root = char(manifest.output_dir);
analyses = manifest.analyses;

software_file = fullfile(output_root, 'SPM_SOFTWARE.txt');
fid = fopen(software_file, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'MATLAB: %s\n', version);
fprintf(fid, 'SPM: %s\n', spm('Ver'));
fprintf(fid, 'SPM directory: %s\n', spm('Dir'));
clear cleanup;

for analysis_index = 1:numel(analyses)
    analysis = analyses(analysis_index);
    participant = char(analysis.participant);
    analysis_name = char(analysis.analysis);
    analysis_kind = char(analysis.kind);
    set_file = char(analysis.set_file);
    tdcm_end_ms = double(analysis.tdcm_end_ms);
    analysis_dir = fullfile(output_root, participant, 'dcm', analysis_name);
    if ~exist(analysis_dir, 'dir')
        mkdir(analysis_dir);
    end

    fprintf('SPM DCM: %s / %s (%s)\n', ...
        participant, analysis_name, analysis_kind);
    spm_file = convert_eeglab_to_spm(set_file, analysis_dir, analysis_name);

    model_files = cell(1, 3);
    free_energy = zeros(1, 3);
    for model_index = 1:3
        model_files{model_index} = fullfile(analysis_dir, ...
            sprintf('DCM_%s_F%d.mat', analysis_name, model_index));
        DCM = configure_dcm( ...
            spm_file, model_files{model_index}, analysis_kind, ...
            model_index, tdcm_end_ms);
        if strcmpi(analysis_kind, 'ERP')
            DCM = spm_dcm_erp_data(DCM);
            DCM = spm_dcm_erp_dipfit(DCM);
            DCM = spm_dcm_erp(DCM);
        elseif strcmpi(analysis_kind, 'CSD')
            DCM = spm_dcm_erp_dipfit(DCM);
            % CSD feature extraction needs channel modes from the dipfit.
            DCM = spm_dcm_csd_data(DCM);
            DCM = spm_dcm_csd(DCM);
        else
            error('Unsupported analysis kind: %s', analysis_kind);
        end
        save(model_files{model_index}, 'DCM', '-v7.3');
        free_energy(model_index) = DCM.F;
    end

    probabilities = softmax_free_energy(free_energy);
    [~, winning_model] = max(probabilities);
    write_model_comparison( ...
        analysis_dir, analysis_name, free_energy, probabilities);
    plot_model_comparison( ...
        analysis_dir, analysis_name, free_energy, probabilities);

    winning = load(model_files{winning_model}, 'DCM');
    connection_table = extract_connection_intervals( ...
        winning.DCM, winning_model);
    writetable(connection_table, ...
        fullfile(analysis_dir, 'winning_model_connections.csv'));
    plot_connections(analysis_dir, analysis_name, winning_model, ...
        probabilities(winning_model), connection_table);

    winner_file = fullfile(analysis_dir, 'WINNING_MODEL.txt');
    fid = fopen(winner_file, 'w');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, 'Winning family/model: F%d\n', winning_model);
    fprintf(fid, 'Posterior probability: %.10f\n', ...
        probabilities(winning_model));
    fprintf(fid, 'Free energy: %.10f\n', free_energy(winning_model));
    fprintf(fid, ['One prespecified model represents each family in this ' ...
        'pilot; probability is therefore model/family probability.\n']);
    clear cleanup;
end
end


function spm_file = convert_eeglab_to_spm(set_file, analysis_dir, analysis_name)
S = struct();
S.dataset = set_file;
S.outfile = fullfile(analysis_dir, ['spm_' analysis_name]);
S.channels = 'all';
S.blocksize = 3276800;
D = spm_eeg_convert(S);
save(D);
spm_file = fullfile(D.path, D.fname);
end


function DCM = configure_dcm( ...
    spm_file, output_file, kind, model_index, tdcm_end_ms)
% Four fixed, hypothesis-driven left language-network source priors (MNI mm).
n = 4;
DCM = struct();
DCM.name = output_file;
DCM.xY.Dfile = spm_file;
DCM.options.spatial = 'ECD';
DCM.options.trials = 1;
DCM.options.D = 1;
DCM.options.Nmodes = 8;
DCM.options.h = 1;
DCM.Sname = {'lOT', 'lpMTG', 'lATL', 'lIFG'};
DCM.Lpos = [ ...
    -42, -56, -50, -48; ...
    -64, -46,   8,  26; ...
    -12,   2, -28,  10];

A = {zeros(n), zeros(n), zeros(n)};
% SPM convention is A{type}(target, source): source -> target.
% F1: OT -> pMTG/ATL -> IFG.
A{1}(2, 1) = 1;
A{1}(3, 1) = 1;
A{1}(4, 2) = 1;
A{1}(4, 3) = 1;
if model_index == 2
    % F2: F1 plus IFG feedback to both temporal nodes.
    A{2}(2, 4) = 1;
    A{2}(3, 4) = 1;
elseif model_index == 3
    % F3: F1 plus a direct OT -> IFG route.
    A{1}(4, 1) = 1;
elseif model_index ~= 1
    error('Unknown model index: %d', model_index);
end
DCM.A = A;
DCM.B = {};
DCM.xU.X = sparse(1, 0);
DCM.xU.name = {};

if strcmpi(kind, 'ERP')
    DCM.options.analysis = 'ERP';
    DCM.options.model = 'ERP';
    DCM.options.Tdcm = [0 tdcm_end_ms];
    DCM.options.Fdcm = [1 30];
    DCM.options.onset = 60;
    DCM.options.dur = 16;
    DCM.C = [1; 0; 0; 0];
elseif strcmpi(kind, 'CSD')
    DCM.options.analysis = 'CSD';
    DCM.options.model = 'CMC';
    DCM.options.Tdcm = [1 tdcm_end_ms];
    DCM.options.Fdcm = [4 40];
    DCM.C = zeros(n, 1);
else
    error('Unsupported analysis kind: %s', kind);
end
end


function probabilities = softmax_free_energy(free_energy)
shifted = free_energy - max(free_energy);
probabilities = exp(shifted);
probabilities = probabilities ./ sum(probabilities);
end


function write_model_comparison(analysis_dir, analysis_name, F, probability)
family = {'F1'; 'F2'; 'F3'};
description = { ...
    'Feedforward OT-temporal-IFG'; ...
    'F1 plus IFG-to-temporal feedback'; ...
    'F1 plus direct OT-to-IFG route'};
free_energy = F(:);
posterior_probability = probability(:);
is_winner = posterior_probability == max(posterior_probability);
results = table(family, description, free_energy, ...
    posterior_probability, is_winner);
writetable(results, fullfile(analysis_dir, ...
    [analysis_name '_model_comparison.csv']));
end


function plot_model_comparison(analysis_dir, analysis_name, F, probability)
figure_handle = figure('Visible', 'off', 'Color', 'white', ...
    'Position', [100 100 1000 420]);
tiledlayout(1, 2, 'Padding', 'compact');
nexttile;
bar(F, 'FaceColor', [0.25 0.45 0.70]);
set(gca, 'XTickLabel', {'F1', 'F2', 'F3'});
ylabel('Variational free energy (log-evidence approximation)');
title('Model evidence');
grid on;
nexttile;
bar(probability, 'FaceColor', [0.75 0.35 0.25]);
set(gca, 'XTickLabel', {'F1', 'F2', 'F3'});
ylim([0 1]);
ylabel('Posterior model/family probability');
title('Fixed-effects posterior probability');
grid on;
sgtitle(strrep(analysis_name, '_', '\_'));
exportgraphics(figure_handle, ...
    fullfile(analysis_dir, [analysis_name '_model_comparison.png']), ...
    'Resolution', 180);
close(figure_handle);
end


function results = extract_connection_intervals(DCM, model_index)
[sources, targets, types, source_index, target_index] = model_edges(model_index);
n_edges = numel(sources);
posterior_mean = zeros(n_edges, 1);
posterior_sd = zeros(n_edges, 1);
ci90_low = zeros(n_edges, 1);
ci90_high = zeros(n_edges, 1);
direction_class = cell(n_edges, 1);

mu = spm_vec(DCM.Ep);
for edge_index = 1:n_edges
    parameter_template = spm_unvec(zeros(size(mu)), DCM.Ep);
    if strcmp(types{edge_index}, 'forward')
        matrix_index = 1;
    else
        matrix_index = 2;
    end
    parameter_template.A{matrix_index}( ...
        target_index(edge_index), source_index(edge_index)) = 1;
    contrast = spm_vec(parameter_template);
    if numel(contrast) ~= size(DCM.Cp, 1)
        error(['Posterior covariance dimensions do not match spm_vec(Ep). ' ...
            'Do not report credible intervals from guessed indices.']);
    end
    posterior_mean(edge_index) = contrast' * mu;
    variance = contrast' * DCM.Cp * contrast;
    posterior_sd(edge_index) = sqrt(max(full(variance), 0));
    ci90_low(edge_index) = posterior_mean(edge_index) ...
        - 1.64485362695147 * posterior_sd(edge_index);
    ci90_high(edge_index) = posterior_mean(edge_index) ...
        + 1.64485362695147 * posterior_sd(edge_index);
    direction_class{edge_index} = types{edge_index};
end

source = sources(:);
target = targets(:);
parameter_label = strcat(source, {' -> '}, target);
results = table(source, target, direction_class, parameter_label, ...
    posterior_mean, posterior_sd, ci90_low, ci90_high);
end


function [sources, targets, types, source_index, target_index] = model_edges(model)
sources = {'lOT', 'lOT', 'lpMTG', 'lATL'};
targets = {'lpMTG', 'lATL', 'lIFG', 'lIFG'};
types = {'forward', 'forward', 'forward', 'forward'};
source_index = [1, 1, 2, 3];
target_index = [2, 3, 4, 4];
if model == 2
    sources = [sources, {'lIFG', 'lIFG'}];
    targets = [targets, {'lpMTG', 'lATL'}];
    types = [types, {'backward', 'backward'}];
    source_index = [source_index, 4, 4];
    target_index = [target_index, 2, 3];
elseif model == 3
    sources = [sources, {'lOT'}];
    targets = [targets, {'lIFG'}];
    types = [types, {'forward'}];
    source_index = [source_index, 1];
    target_index = [target_index, 4];
end
end


function plot_connections(analysis_dir, analysis_name, winning_model, ...
    winner_probability, results)
figure_handle = figure('Visible', 'off', 'Color', 'white', ...
    'Position', [100 100 1200 520]);
tiledlayout(1, 2, 'Padding', 'compact');

nexttile;
sources = cellstr(results.source);
targets = cellstr(results.target);
edge_labels = compose('%.2f', results.posterior_mean);
graph_object = digraph(sources, targets);
graph_plot = plot(graph_object, 'Layout', 'layered', ...
    'Direction', 'right', 'EdgeLabel', edge_labels, ...
    'LineWidth', 1.5, 'ArrowSize', 12, 'NodeFontSize', 10);
graph_plot.NodeColor = [0.15 0.35 0.65];
graph_plot.EdgeColor = [0.25 0.25 0.25];
title(sprintf('Winning F%d network (p=%.3f)', ...
    winning_model, winner_probability));
axis off;

nexttile;
means = results.posterior_mean;
low_error = means - results.ci90_low;
high_error = results.ci90_high - means;
barh(means, 'FaceColor', [0.35 0.55 0.75]);
hold on;
for index = 1:height(results)
    line([means(index) - low_error(index), ...
        means(index) + high_error(index)], [index index], ...
        'Color', 'black', 'LineWidth', 1.5);
    plot(means(index), index, 'ko', 'MarkerFaceColor', 'black', ...
        'MarkerSize', 4);
end
hold off;
set(gca, 'YTick', 1:height(results), ...
    'YTickLabel', cellstr(results.parameter_label));
xline(0, '--', 'Color', [0.4 0.4 0.4]);
xlabel('Posterior parameter (SPM native units)');
title('Connection posterior means and 90% credible intervals');
grid on;

sgtitle(strrep(analysis_name, '_', '\_'));
exportgraphics(figure_handle, ...
    fullfile(analysis_dir, [analysis_name '_winning_dcm.png']), ...
    'Resolution', 180);
close(figure_handle);
end
