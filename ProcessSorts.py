from sfsortingresults import SFSortingResults
import numpy as np
import spikeextractors as se

import os
import yaml
import glob2

# make required files/directories
class ProcessSorts():

    def __init__(self, fname_yass, sfresults):

        #
        self.root_dir = '/media/cat/1TB/spikesorting/sorting_analysis/'

        #
        self.sfresults = sfresults

        #
        self.data_yass = np.load(fname_yass)
        #self.target_folder = self.data_yass['folder_name']
        self.target_study = self.data_yass['study_name']
        self.target_recording = self.data_yass['rec_name']
        # print (self.target_folder, self.target_study, self.target_recording)

        # get config file name
        fname_config = fname_yass[:-4]+".yaml"
        with open(fname_config) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        #
        self.make_dirs()


    def extract_all_sorts(self):

        #
        self.save_yass()

        #
        self.save_sorts()

        #
        self.save_gt()


    def make_dirs(self):

        self.dir_sortings = os.path.join(self.root_dir,
                            str(self.target_study),
                            str(self.target_recording),'sortings')
        self.dir_run_log = os.path.join(self.root_dir,
                            str(self.target_study),
                            str(self.target_recording),'sortings','run_log')
        self.dir_ground_truth = os.path.join(self.root_dir,
                            str(self.target_study),
                            str(self.target_recording),'ground_truth')
        self.dir_yass = os.path.join(self.root_dir,
                            str(self.target_study),
                            str(self.target_recording),'yass')
        self.dir_yass_output = os.path.join(self.root_dir,
                            str(self.target_study),
                            str(self.target_recording),'yass','tmp','output')
#
        if os.path.exists(self.dir_sortings)==False:
            os.makedirs(self.dir_sortings)
        if os.path.exists(self.dir_ground_truth)==False:
            os.makedirs(self.dir_ground_truth)
        if os.path.exists(self.dir_yass_output)==False:
            os.makedirs(self.dir_yass_output)
        if os.path.exists(self.dir_run_log)==False:
            os.makedirs(self.dir_run_log)

        # save names.txt
        np.savetxt(os.path.join(self.root_dir,
                            str(self.target_study),
                            str(self.target_recording),
                            'names.txt'),['rec0'], fmt="%s")
        #num.savetxt('test.txt', DAT, delimiter=" ", fmt="%s")

    #
    def save_yass(self):

        #
        spike_train = self.data_yass['spike_train']
        templates = self.data_yass['templates']

        #
        print (self.root_dir)
        with open(self.dir_yass+'/config.yaml', 'w') as file:
            documents = yaml.dump(self.config, file)

        # remove single spike results from Yass
        spike_train_new = np.zeros((0,2),'int32')
        ctr=0
        for k in np.unique(spike_train[:,1]):
            idx = np.where(spike_train[:,1]==k)[0]
            if idx.shape[0]>1:
                times = spike_train[idx,0]
                ids = times*0 + ctr

                temp = np.zeros((times.shape[0],2),'int32')
                temp[:,0] = times
                temp[:,1] = ids

                spike_train_new = np.vstack((spike_train_new, temp))

                ctr+=1

        spike_train = spike_train_new
        #
        np.save(self.dir_yass_output+"/spike_train.npy", spike_train)
        np.save(self.dir_yass_output+"/templates.npy", templates)

        # load sorting object
        yass_sort = se.YassSortingExtractor(self.dir_yass)

        #
        save_path = os.path.join(self.dir_sortings, "rec0[#]yass.npz")
        se.NpzSortingExtractor.write_sorting(yass_sort, save_path)


    def save_sorts(self):
        # retrieve available sorter outputs
        sorting_output_names = self.sfresults.get_sorting_output_names(study_name=self.target_study,
                                                                  recording_name=self.target_recording)
        print(f"Recording {self.target_recording} has the following sorting outputs:\n{sorting_output_names}")

        #
        for target_sorting_output in sorting_output_names:
            print ("sorter selected: ", target_sorting_output)
            self.target_sorting_output = target_sorting_output
            try:
                sorting = self.sfresults.get_sorting_output(study_name=self.target_study,
                                                   recording_name=self.target_recording,
                                                   sorter_name=target_sorting_output)
                self.save_sorter(sorting)
            except Exception as e:
                print ("error: ", e)
            print ('')
        # run comparison study



    #
    def save_sorter(self, sorting):

        # for all sorters
        save_path = os.path.join(self.dir_sortings, "rec0[#]"+self.target_sorting_output+'.npz')
        se.NpzSortingExtractor.write_sorting(sorting, save_path)



    def save_gt(self):

        gt_sorting = self.sfresults.get_gt_sorting_output(study_name=self.target_study,
                                                     recording_name=self.target_recording)

        save_path = os.path.join(self.dir_ground_truth, 'rec0.npz')
        se.NpzSortingExtractor.write_sorting(gt_sorting, save_path)


    def run_GTstudy(self):
        study_folder = '/media/cat/1TB/spikesorting/sorting_analysis/hybrid_static_siprobe/rec_32c_600s_11/'
        study = GroundTruthStudy(study_folder)

        # fix the ground truth sampling frequency
        fnames = glob2.glob(study_folder+ 'sortings/*.npz')
        sf = np.load(fnames[0])['sampling_frequency']

        gt_data = np.load(study_folder+'/ground_truth/rec0.npz')
        np.savez(study_folder+'/ground_truth/rec0.npz',
                unit_ids = gt_data['unit_ids'],
                spike_labels = gt_data['spike_labels'],
                spike_indexes = gt_data['spike_indexes'],
                sampling_frequency = sf)


        #
        study.run_comparisons(exhaustive_gt=True,
                              match_score=0.0,
                              delta_time=3.0)
